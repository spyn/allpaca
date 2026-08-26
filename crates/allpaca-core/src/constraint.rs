use allpaca_types::TokenMask;

/// Hardcoded JSON grammar for `json_mode` without a schema.
/// Schema and Lark requests go through `grammar` / llguidance.
#[derive(Debug, Clone)]
pub struct JsonConstraint {
    vocab: Vec<String>,
}

impl JsonConstraint {
    pub fn new(vocab: Vec<String>) -> Self {
        Self { vocab }
    }

    pub fn mask(&self, prefix: &str) -> TokenMask {
        let mut mask = TokenMask::empty(self.vocab.len());
        for (id, tok) in self.vocab.iter().enumerate() {
            if token_allowed(prefix, tok) {
                mask.set(id, true);
            }
        }
        mask
    }
}

fn token_allowed(prefix: &str, token: &str) -> bool {
    if is_special_stop(token) {
        return matches!(scan_json_prefix(prefix), Ok(JsonScan::Complete));
    }
    scan_json_prefix(&format!("{prefix}{token}")).is_ok()
}

fn is_special_stop(token: &str) -> bool {
    matches!(
        token,
        "</s>" | "<s>" | "<unk>" | "<|endoftext|>" | "<|eot_id|>"
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonScan {
    Incomplete,
    Complete,
}

fn scan_json_prefix(s: &str) -> Result<JsonScan, ()> {
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    skip_ws(&chars, &mut i);
    if i >= chars.len() {
        return Ok(JsonScan::Incomplete);
    }
    let complete = parse_value(&chars, &mut i)?;
    skip_ws(&chars, &mut i);
    if i < chars.len() {
        return Err(());
    }
    if complete {
        Ok(JsonScan::Complete)
    } else {
        Ok(JsonScan::Incomplete)
    }
}

fn skip_ws(chars: &[char], i: &mut usize) {
    while *i < chars.len() && chars[*i].is_whitespace() {
        *i += 1;
    }
}

fn parse_value(chars: &[char], i: &mut usize) -> Result<bool, ()> {
    if *i >= chars.len() {
        return Ok(false);
    }
    match chars[*i] {
        '{' => parse_object(chars, i),
        '[' => parse_array(chars, i),
        '"' => parse_string(chars, i),
        't' => parse_literal(chars, i, "true"),
        'f' => parse_literal(chars, i, "false"),
        'n' => parse_literal(chars, i, "null"),
        '-' | '0'..='9' => parse_number(chars, i),
        _ => Err(()),
    }
}

fn parse_object(chars: &[char], i: &mut usize) -> Result<bool, ()> {
    *i += 1;
    skip_ws(chars, i);
    if *i >= chars.len() {
        return Ok(false);
    }
    if chars[*i] == '}' {
        *i += 1;
        return Ok(true);
    }
    loop {
        skip_ws(chars, i);
        if *i >= chars.len() {
            return Ok(false);
        }
        if chars[*i] != '"' {
            return Err(());
        }
        if !parse_string(chars, i)? {
            return Ok(false);
        }
        skip_ws(chars, i);
        if *i >= chars.len() {
            return Ok(false);
        }
        if chars[*i] != ':' {
            return Err(());
        }
        *i += 1;
        skip_ws(chars, i);
        if *i >= chars.len() {
            return Ok(false);
        }
        if !parse_value(chars, i)? {
            return Ok(false);
        }
        skip_ws(chars, i);
        if *i >= chars.len() {
            return Ok(false);
        }
        match chars[*i] {
            ',' => {
                *i += 1;
            }
            '}' => {
                *i += 1;
                return Ok(true);
            }
            _ => return Err(()),
        }
    }
}

fn parse_array(chars: &[char], i: &mut usize) -> Result<bool, ()> {
    *i += 1;
    skip_ws(chars, i);
    if *i >= chars.len() {
        return Ok(false);
    }
    if chars[*i] == ']' {
        *i += 1;
        return Ok(true);
    }
    loop {
        skip_ws(chars, i);
        if *i >= chars.len() {
            return Ok(false);
        }
        if !parse_value(chars, i)? {
            return Ok(false);
        }
        skip_ws(chars, i);
        if *i >= chars.len() {
            return Ok(false);
        }
        match chars[*i] {
            ',' => {
                *i += 1;
            }
            ']' => {
                *i += 1;
                return Ok(true);
            }
            _ => return Err(()),
        }
    }
}

fn parse_string(chars: &[char], i: &mut usize) -> Result<bool, ()> {
    *i += 1;
    while *i < chars.len() {
        match chars[*i] {
            '"' => {
                *i += 1;
                return Ok(true);
            }
            '\\' => {
                *i += 1;
                if *i >= chars.len() {
                    return Ok(false);
                }
                *i += 1;
            }
            c if c.is_control() => return Err(()),
            _ => *i += 1,
        }
    }
    Ok(false)
}

fn parse_literal(chars: &[char], i: &mut usize, lit: &str) -> Result<bool, ()> {
    for (idx, expected) in lit.chars().enumerate() {
        if *i >= chars.len() {
            return if idx == 0 { Err(()) } else { Ok(false) };
        }
        if chars[*i] != expected {
            return Err(());
        }
        *i += 1;
    }
    Ok(true)
}

fn parse_number(chars: &[char], i: &mut usize) -> Result<bool, ()> {
    if chars[*i] == '-' {
        *i += 1;
        if *i >= chars.len() {
            return Ok(false);
        }
    }
    if *i >= chars.len() || !chars[*i].is_ascii_digit() {
        return Err(());
    }
    if chars[*i] == '0' {
        *i += 1;
    } else {
        while *i < chars.len() && chars[*i].is_ascii_digit() {
            *i += 1;
        }
    }
    if *i < chars.len() && chars[*i] == '.' {
        *i += 1;
        if *i >= chars.len() {
            return Ok(false);
        }
        if !chars[*i].is_ascii_digit() {
            return Err(());
        }
        while *i < chars.len() && chars[*i].is_ascii_digit() {
            *i += 1;
        }
    }
    if *i < chars.len() && (chars[*i] == 'e' || chars[*i] == 'E') {
        *i += 1;
        if *i < chars.len() && (chars[*i] == '+' || chars[*i] == '-') {
            *i += 1;
        }
        if *i >= chars.len() {
            return Ok(false);
        }
        if !chars[*i].is_ascii_digit() {
            return Err(());
        }
        while *i < chars.len() && chars[*i].is_ascii_digit() {
            *i += 1;
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_prefix_allows_object() {
        let c = JsonConstraint::new(vec!["{".into(), "x".into(), "}".into()]);
        let mask = c.mask("");
        assert!(mask.allowed(0));
        assert!(!mask.allowed(1));
    }

    #[test]
    fn complete_object_rejects_extra_junk() {
        assert!(scan_json_prefix("{\"a\":1}").is_ok());
        assert!(scan_json_prefix("{\"a\":1}x").is_err());
        assert!(scan_json_prefix("{").is_ok());
    }
}
