//! Chat `image_url` loader: data URIs and public http(s) only.
//!
//! Filesystem paths and `file://` are rejected. HTTP(S) fetches do not follow
//! redirects and refuse loopback, private, link-local, and other non-global
//! addresses (including after DNS resolution).

use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};
use std::time::Duration;

use allpaca_core::ImageInput;
use allpaca_types::{AllpacaError, FatalError};

/// Max decoded / downloaded image size (8 MiB).
pub const MAX_IMAGE_BYTES: usize = 8 * 1024 * 1024;
const FETCH_TIMEOUT: Duration = Duration::from_secs(30);

pub fn load_image_url(url: &str) -> Result<ImageInput, AllpacaError> {
    let raw = url.trim();
    if raw.is_empty() {
        return Err(denied("empty image_url"));
    }
    if raw
        .get(..5)
        .is_some_and(|s| s.eq_ignore_ascii_case("data:"))
    {
        return load_data_uri(raw);
    }
    let scheme = scheme_of(raw).unwrap_or("");
    if scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https") {
        return fetch_http_image(raw);
    }
    Err(denied(
        "image_url must be a data: URI or http(s) URL (file paths are not allowed)",
    ))
}

fn denied(msg: &str) -> AllpacaError {
    AllpacaError::Fatal(FatalError::InvalidConfig(msg.into()))
}

fn scheme_of(raw: &str) -> Option<&str> {
    let i = raw.find("://")?;
    (i > 0).then_some(&raw[..i])
}

fn load_data_uri(url: &str) -> Result<ImageInput, AllpacaError> {
    let (_, after_scheme) = url
        .split_once(':')
        .ok_or_else(|| denied("invalid data URI"))?;
    let (meta, payload) = after_scheme
        .split_once(',')
        .ok_or_else(|| denied("data URI missing comma separator"))?;
    let mut mime = "application/octet-stream".to_string();
    let mut is_b64 = false;
    for part in meta.split(';').filter(|p| !p.is_empty()) {
        if part.eq_ignore_ascii_case("base64") {
            is_b64 = true;
        } else if part.contains('/') {
            mime = part.to_ascii_lowercase();
        }
    }
    if payload.len() > MAX_IMAGE_BYTES.saturating_mul(2) {
        return Err(denied("image_url exceeds size limit"));
    }
    let bytes = if is_b64 {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD
            .decode(payload.trim().as_bytes())
            .map_err(|e| {
                AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                    "invalid data URI base64: {e}"
                )))
            })?
    } else {
        payload.as_bytes().to_vec()
    };
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(denied("image_url exceeds size limit"));
    }
    Ok(ImageInput::new(bytes, mime))
}

fn fetch_http_image(url: &str) -> Result<ImageInput, AllpacaError> {
    let parsed = parse_public_http_url(url)?;
    let client = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(FETCH_TIMEOUT)
        .user_agent("allpaca/0.1")
        .build()
        .map_err(|e| AllpacaError::Fatal(FatalError::Io(format!("http client init: {e}"))))?;
    let mut resp = client
        .get(parsed.as_str())
        .send()
        .map_err(|e| AllpacaError::Fatal(FatalError::Io(format!("fetch image_url: {e}"))))?;
    if !resp.status().is_success() {
        return Err(AllpacaError::Fatal(FatalError::Io(format!(
            "fetch image_url HTTP {}",
            resp.status()
        ))));
    }
    if let Some(len) = resp.content_length() {
        if len > MAX_IMAGE_BYTES as u64 {
            return Err(denied("image_url exceeds size limit"));
        }
    }
    let mime = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.split(';').next().unwrap_or(s).trim().to_ascii_lowercase())
        .filter(|s| s.starts_with("image/"))
        .unwrap_or_else(|| mime_from_url_path(parsed.path()));
    let mut buf = Vec::new();
    let n = Read::take(&mut resp, MAX_IMAGE_BYTES as u64 + 1)
        .read_to_end(&mut buf)
        .map_err(|e| AllpacaError::Fatal(FatalError::Io(format!("read image_url body: {e}"))))?;
    if n > MAX_IMAGE_BYTES {
        return Err(denied("image_url exceeds size limit"));
    }
    Ok(ImageInput::new(buf, mime))
}

fn parse_public_http_url(url: &str) -> Result<reqwest::Url, AllpacaError> {
    let parsed = reqwest::Url::parse(url).map_err(|_| denied("invalid image_url"))?;
    let scheme = parsed.scheme();
    if scheme != "http" && scheme != "https" {
        return Err(denied("image_url must be http or https"));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(denied("image_url must not include credentials"));
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| denied("image_url missing host"))?;
    if let Ok(ip) = host.parse::<IpAddr>() {
        if ip_blocked(ip) {
            return Err(denied("image_url host is not allowed"));
        }
        return Ok(parsed);
    }
    if domain_blocked(host) {
        return Err(denied("image_url host is not allowed"));
    }
    let port = parsed.port_or_known_default().unwrap_or(80);
    let addrs = (host, port)
        .to_socket_addrs()
        .map_err(|_| denied("image_url host is not allowed"))?;
    let mut any = false;
    for addr in addrs {
        any = true;
        if ip_blocked(addr.ip()) {
            return Err(denied("image_url host is not allowed"));
        }
    }
    if !any {
        return Err(denied("image_url host is not allowed"));
    }
    Ok(parsed)
}

fn domain_blocked(name: &str) -> bool {
    let n = name.trim_end_matches('.').to_ascii_lowercase();
    n.is_empty()
        || n == "localhost"
        || n.ends_with(".localhost")
        || n == "metadata.google.internal"
        || n.ends_with(".internal")
}

fn ip_blocked(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => ipv4_blocked(v4),
        IpAddr::V6(v6) => ipv6_blocked(v6),
    }
}

fn ipv4_blocked(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_multicast()
        || ip.is_documentation()
        || o[0] == 0
        || o[0] >= 240
        || (o[0] == 100 && o[1] >= 64 && o[1] <= 127)
        || (o[0] == 198 && (o[1] == 18 || o[1] == 19))
}

fn ipv6_blocked(ip: Ipv6Addr) -> bool {
    if ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_multicast()
        || ip.is_unicast_link_local()
        || ip.is_unique_local()
    {
        return true;
    }
    if let Some(v4) = ip.to_ipv4_mapped() {
        return ipv4_blocked(v4);
    }
    if let Some(v4) = ip.to_ipv4() {
        return ipv4_blocked(v4);
    }
    false
}

fn mime_from_url_path(path: &str) -> String {
    match path
        .rsplit('.')
        .next()
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => "image/png".into(),
        Some("gif") => "image/gif".into(),
        Some("webp") => "image/webp".into(),
        Some("avif") => "image/avif".into(),
        Some("jpg") | Some("jpeg") => "image/jpeg".into(),
        _ => "application/octet-stream".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err_msg(url: &str) -> String {
        load_image_url(url).expect_err("must reject").to_string()
    }

    #[test]
    fn rejects_filesystem_path() {
        let msg = err_msg("/etc/passwd");
        assert!(msg.contains("file paths are not allowed"), "{msg}");
    }

    #[test]
    fn rejects_file_scheme() {
        let msg = err_msg("file:///etc/passwd");
        assert!(msg.contains("file paths are not allowed"), "{msg}");
        let msg = err_msg("FILE://localhost/etc/passwd");
        assert!(msg.contains("file paths are not allowed"), "{msg}");
    }

    #[test]
    fn rejects_loopback_and_private_literals() {
        for url in [
            "http://127.0.0.1/",
            "http://127.0.0.1:8080/secret",
            "https://localhost/img.png",
            "http://[::1]/",
            "http://10.0.0.1/x",
            "http://192.168.1.1/x",
            "http://172.16.0.1/x",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::ffff:127.0.0.1]/",
            "http://100.64.0.1/x",
            "http://0.0.0.0/",
        ] {
            let msg = err_msg(url);
            assert!(
                msg.contains("not allowed") || msg.contains("file paths"),
                "{url} => {msg}"
            );
        }
    }

    #[test]
    fn rejects_userinfo() {
        let msg = err_msg("https://user:pass@example.com/a.png");
        assert!(msg.contains("credentials"), "{msg}");
    }

    #[test]
    fn data_uri_still_loads() {
        let raw = [0xFFu8, 0xD8, 0xFF, 0xD9];
        let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, raw);
        let img = load_image_url(&format!("data:image/jpeg;base64,{b64}")).unwrap();
        assert_eq!(img.mime, "image/jpeg");
        assert_eq!(img.bytes, raw);
    }

    #[test]
    fn data_uri_rejects_oversize() {
        let payload = "A".repeat(MAX_IMAGE_BYTES.saturating_mul(3));
        let url = format!("data:image/jpeg;base64,{payload}");
        let msg = err_msg(&url);
        assert!(msg.contains("size limit"), "{msg}");
    }
}
