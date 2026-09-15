//! HuggingFace Transformers tensor names → engine (GGUF-style) lookup lists.
//!
//! Dense Llama / Qwen2 / Mistral / Gemma checkpoints store Linear weights as
//! `[out, in]` row-major, matching the engine's NT GEMM layout.

/// One weight the dense transformer loader needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightKey {
    Embed,
    OutputNorm,
    LmHead,
    AttnNorm(usize),
    QNorm(usize),
    KNorm(usize),
    Wq(usize),
    Wk(usize),
    Wv(usize),
    Wo(usize),
    FfnNorm(usize),
    Gate(usize),
    Up(usize),
    Down(usize),
}

/// Candidate tensor names, GGUF-style first so existing fixtures keep working.
pub fn aliases(key: WeightKey) -> Vec<String> {
    match key {
        WeightKey::Embed => vec![
            "token_embd.weight".into(),
            "model.embed_tokens.weight".into(),
            "embed_tokens.weight".into(),
            "model.model.embed_tokens.weight".into(),
        ],
        WeightKey::OutputNorm => vec![
            "output_norm.weight".into(),
            "model.norm.weight".into(),
            "model.model.norm.weight".into(),
            "model.final_layernorm.weight".into(),
        ],
        WeightKey::LmHead => vec!["output.weight".into(), "lm_head.weight".into()],
        WeightKey::AttnNorm(i) => vec![
            format!("blk.{i}.attn_norm.weight"),
            format!("model.layers.{i}.input_layernorm.weight"),
            format!("layers.{i}.input_layernorm.weight"),
            format!("model.layers.{i}.rms_1.weight"),
        ],
        WeightKey::QNorm(i) => vec![
            format!("blk.{i}.attn_q_norm.weight"),
            format!("model.layers.{i}.self_attn.q_norm.weight"),
            format!("model.layers.{i}.self_attn.q_layernorm.weight"),
        ],
        WeightKey::KNorm(i) => vec![
            format!("blk.{i}.attn_k_norm.weight"),
            format!("model.layers.{i}.self_attn.k_norm.weight"),
            format!("model.layers.{i}.self_attn.k_layernorm.weight"),
        ],
        WeightKey::Wq(i) => vec![
            format!("blk.{i}.attn_q.weight"),
            format!("model.layers.{i}.self_attn.q_proj.weight"),
            format!("layers.{i}.self_attn.q_proj.weight"),
        ],
        WeightKey::Wk(i) => vec![
            format!("blk.{i}.attn_k.weight"),
            format!("model.layers.{i}.self_attn.k_proj.weight"),
            format!("layers.{i}.self_attn.k_proj.weight"),
        ],
        WeightKey::Wv(i) => vec![
            format!("blk.{i}.attn_v.weight"),
            format!("model.layers.{i}.self_attn.v_proj.weight"),
            format!("layers.{i}.self_attn.v_proj.weight"),
        ],
        WeightKey::Wo(i) => vec![
            format!("blk.{i}.attn_output.weight"),
            format!("model.layers.{i}.self_attn.o_proj.weight"),
            format!("layers.{i}.self_attn.o_proj.weight"),
        ],
        WeightKey::FfnNorm(i) => vec![
            format!("blk.{i}.ffn_norm.weight"),
            format!("model.layers.{i}.post_attention_layernorm.weight"),
            format!("layers.{i}.post_attention_layernorm.weight"),
            format!("model.layers.{i}.rms_2.weight"),
        ],
        WeightKey::Gate(i) => vec![
            format!("blk.{i}.ffn_gate.weight"),
            format!("model.layers.{i}.mlp.gate_proj.weight"),
            format!("layers.{i}.mlp.gate_proj.weight"),
            format!("model.layers.{i}.mlp.w1.weight"),
        ],
        WeightKey::Up(i) => vec![
            format!("blk.{i}.ffn_up.weight"),
            format!("model.layers.{i}.mlp.up_proj.weight"),
            format!("layers.{i}.mlp.up_proj.weight"),
            format!("model.layers.{i}.mlp.w3.weight"),
        ],
        WeightKey::Down(i) => vec![
            format!("blk.{i}.ffn_down.weight"),
            format!("model.layers.{i}.mlp.down_proj.weight"),
            format!("layers.{i}.mlp.down_proj.weight"),
            format!("model.layers.{i}.mlp.w2.weight"),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_hf_q_proj_is_listed_for_layer() {
        let names = aliases(WeightKey::Wq(3));
        assert!(names.contains(&"blk.3.attn_q.weight".into()));
        assert!(names.contains(&"model.layers.3.self_attn.q_proj.weight".into()));
    }
}
