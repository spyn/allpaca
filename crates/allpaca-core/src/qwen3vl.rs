//! Qwen3-VL vision tower loaded from a llama.cpp `mmproj` GGUF
//! (`clip.projector_type = qwen3vl_merger`, e.g. the Qwen3.5 / Qwen3.8 mmproj).
//!
//! Mirrors llama.cpp `clip_graph_qwen3vl`: both temporal patch kernels applied
//! to one still frame (summed at load), patches in 2x2 spatial-merge order, a
//! bilinear (align-corners) learned position embedding, pre-norm ViT blocks
//! with LayerNorm, vision M-RoPE over (row, column) and tanh GELU, a final
//! LayerNorm, then each 2x2 group concatenated through a GELU MLP to the LLM
//! width. Deepstack layers are rejected (the Qwen3.5 / 3.8 mmproj has none).
//! Large matrices stay F16 on the host (as in the file). [`Qwen3VlVision::encode`]
//! is the CPU reference; [`Qwen3VlVision::encode_on`] runs the same graph on a
//! HIP device, uploading the weights for the call and freeing them after, so
//! the encoder never holds VRAM the LLM's layer plan did not budget for.

use std::path::Path;

use allpaca_gguf::{GgufFile, GgufValue};
use allpaca_types::{AllpacaError, FatalError};

const PROJECTOR: &str = "qwen3vl_merger";
const ROPE_THETA: f32 = 10_000.0;
/// Fewest merged tokens per image (llama.cpp Qwen-VL limit).
const MIN_IMAGE_TOKENS: usize = 8;
/// Default token budget per image. llama.cpp allows 4096 (~2048 px square);
/// 1024 (~1024 px square) keeps encode time reasonable and is what llama.cpp
/// recommends as the floor for grounding.
pub const DEFAULT_MAX_IMAGE_TOKENS: usize = 1024;

/// Shape of a Qwen3-VL vision tower.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3VlConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub n_ff: usize,
    pub patch_size: usize,
    /// Patches per side merged into one LLM token.
    pub merge: usize,
    pub eps: f32,
    /// LLM hidden size the projector writes.
    pub proj_dim: usize,
    /// Side of the learned position-embedding grid.
    pub pos_side: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl Qwen3VlConfig {
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

/// Weight matrices are F16 bits `[out][in]`; norms and biases F32.
struct Block {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    qkv_w: Vec<u16>,
    qkv_b: Vec<f32>,
    out_w: Vec<u16>,
    out_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    up_w: Vec<u16>,
    up_b: Vec<f32>,
    down_w: Vec<u16>,
    down_b: Vec<f32>,
}

/// Qwen3-VL vision encoder with F32 host weights.
pub struct Qwen3VlVision {
    cfg: Qwen3VlConfig,
    /// `[n_embd][3 * patch * patch]`, both temporal kernels summed.
    patch_w: Vec<f32>,
    patch_b: Vec<f32>,
    /// `[pos_side * pos_side][n_embd]`, row-major grid.
    pos: Vec<f32>,
    blocks: Vec<Block>,
    post_w: Vec<f32>,
    post_b: Vec<f32>,
    mm0_w: Vec<u16>,
    mm0_b: Vec<f32>,
    mm2_w: Vec<u16>,
    mm2_b: Vec<f32>,
    max_image_tokens: usize,
    /// Fixed HIP device for encode (`None`: the GPU with most free memory).
    hip_device: Option<i32>,
    #[allow(clippy::type_complexity)]
    hip: std::sync::Mutex<Vec<(i32, std::sync::Arc<allpaca_hip::HipWorkspace>)>>,
}

fn load_err(msg: impl Into<String>) -> AllpacaError {
    AllpacaError::Fatal(FatalError::ModelLoad(msg.into()))
}

fn meta_array<'a>(file: &'a GgufFile, key: &str) -> Option<&'a [GgufValue]> {
    match file.metadata.get(key)? {
        GgufValue::Array(v) => Some(v.as_slice()),
        _ => None,
    }
}

impl Qwen3VlVision {
    /// Load an mmproj GGUF whose projector is `qwen3vl_merger`.
    pub fn load_gguf(path: &Path) -> Result<Self, AllpacaError> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    pub fn from_gguf(file: &GgufFile) -> Result<Self, AllpacaError> {
        let proj = file.meta_str("clip.projector_type").unwrap_or("");
        if proj != PROJECTOR {
            return Err(load_err(format!("mmproj projector {proj:?} is not {PROJECTOR}")));
        }
        let deepstack = meta_array(file, "clip.vision.is_deepstack_layers")
            .is_some_and(|v| v.iter().any(|b| matches!(b, GgufValue::Bool(true))));
        if deepstack {
            return Err(load_err("qwen3vl mmproj deepstack layers are not supported"));
        }
        let u = |k: &str| {
            file.meta_u32(&format!("clip.vision.{k}"))
                .map(|v| v as usize)
                .ok_or_else(|| load_err(format!("mmproj missing clip.vision.{k}")))
        };
        let triple = |k: &str| {
            let v = meta_array(file, &format!("clip.vision.{k}"))?;
            let f = |x: &GgufValue| match x {
                GgufValue::Float32(f) => Some(*f),
                _ => None,
            };
            Some([f(v.first()?)?, f(v.get(1)?)?, f(v.get(2)?)?])
        };
        let n_embd = u("embedding_length")?;
        let pos_rows = file
            .tensor("v.position_embd.weight")
            .map_or(0, |t| t.n_elements() as usize / n_embd.max(1));
        let pos_side = (pos_rows as f64).sqrt().round() as usize;
        if pos_side == 0 || pos_side * pos_side != pos_rows {
            return Err(load_err(format!("mmproj position grid of {pos_rows} is not square")));
        }
        let cfg = Qwen3VlConfig {
            n_embd,
            n_head: u("attention.head_count")?,
            n_layer: u("block_count")?,
            n_ff: u("feed_forward_length")?,
            patch_size: u("patch_size")?,
            merge: file
                .meta_u32("clip.vision.spatial_merge_size")
                .map_or(2, |v| v as usize),
            eps: file
                .meta_f32("clip.vision.attention.layer_norm_epsilon")
                .unwrap_or(1e-6),
            proj_dim: u("projection_dim")?,
            pos_side,
            image_mean: triple("image_mean").unwrap_or([0.5; 3]),
            image_std: triple("image_std").unwrap_or([0.5; 3]),
        };
        if cfg.merge != 2 || cfg.n_head == 0 || n_embd % cfg.n_head != 0 || cfg.head_dim() % 4 != 0 {
            return Err(load_err(format!("unsupported qwen3vl mmproj shape {cfg:?}")));
        }
        let t = |name: &str| file.tensor_f32(name);
        let f16 = |name: &str| -> Result<Vec<u16>, AllpacaError> {
            let info = file
                .tensor(name)
                .ok_or_else(|| load_err(format!("mmproj missing {name}")))?;
            if info.ggml_type == allpaca_gguf::GgmlType::F16 {
                let bytes = file.tensor_bytes(name)?;
                Ok(bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect())
            } else {
                Ok(t(name)?.into_iter().map(allpaca_cpu::f32_to_f16).collect())
            }
        };
        let mut patch_w = t("v.patch_embd.weight")?;
        if file.tensor("v.patch_embd.weight.1").is_some() {
            // A still image fills both temporal slots, so the two kernels add.
            for (w, w1) in patch_w.iter_mut().zip(t("v.patch_embd.weight.1")?) {
                *w += w1;
            }
        }
        let blocks = (0..cfg.n_layer)
            .map(|i| {
                let b = |s: &str| t(&format!("v.blk.{i}.{s}"));
                let w = |s: &str| f16(&format!("v.blk.{i}.{s}"));
                Ok(Block {
                    ln1_w: b("ln1.weight")?,
                    ln1_b: b("ln1.bias")?,
                    qkv_w: w("attn_qkv.weight")?,
                    qkv_b: b("attn_qkv.bias")?,
                    out_w: w("attn_out.weight")?,
                    out_b: b("attn_out.bias")?,
                    ln2_w: b("ln2.weight")?,
                    ln2_b: b("ln2.bias")?,
                    up_w: w("ffn_up.weight")?,
                    up_b: b("ffn_up.bias")?,
                    down_w: w("ffn_down.weight")?,
                    down_b: b("ffn_down.bias")?,
                })
            })
            .collect::<Result<Vec<_>, AllpacaError>>()?;
        let vision = Self {
            patch_w,
            patch_b: t("v.patch_embd.bias")?,
            pos: t("v.position_embd.weight")?,
            blocks,
            post_w: t("v.post_ln.weight")?,
            post_b: t("v.post_ln.bias")?,
            mm0_w: f16("mm.0.weight")?,
            mm0_b: t("mm.0.bias")?,
            mm2_w: f16("mm.2.weight")?,
            mm2_b: t("mm.2.bias")?,
            max_image_tokens: DEFAULT_MAX_IMAGE_TOKENS,
            hip_device: None,
            hip: std::sync::Mutex::new(Vec::new()),
            cfg,
        };
        vision.check_shapes()?;
        Ok(vision)
    }

    fn check_shapes(&self) -> Result<(), AllpacaError> {
        let c = &self.cfg;
        let (e, pp, merged) = (c.n_embd, 3 * c.patch_size * c.patch_size, 4 * c.n_embd);
        let ok = self.patch_w.len() == e * pp
            && self.patch_b.len() == e
            && self.pos.len() == c.pos_side * c.pos_side * e
            && self.mm0_w.len() == merged * merged
            && self.mm2_w.len() == c.proj_dim * merged
            && self.blocks.iter().all(|b| {
                b.qkv_w.len() == 3 * e * e
                    && b.out_w.len() == e * e
                    && b.up_w.len() == c.n_ff * e
                    && b.down_w.len() == e * c.n_ff
            });
        if ok {
            Ok(())
        } else {
            Err(load_err("qwen3vl mmproj tensor shapes do not match its metadata"))
        }
    }

    pub fn config(&self) -> &Qwen3VlConfig {
        &self.cfg
    }

    /// Cap each image at `max` merged tokens (at least the llama.cpp minimum).
    pub fn with_max_image_tokens(mut self, max: usize) -> Self {
        self.max_image_tokens = max.max(MIN_IMAGE_TOKENS);
        self
    }

    /// Resize (bilinear, aspect preserved, sides multiples of `patch * merge`,
    /// within the token budget) and normalize to planar RGB, as llama.cpp's
    /// Qwen-VL `dyn_size` preprocessor does. Returns `(pixels, width, height)`.
    pub fn preprocess(&self, img: &image::RgbImage) -> (Vec<f32>, usize, usize) {
        let c = &self.cfg;
        let unit = c.patch_size * c.merge;
        let area = unit * unit;
        let (w, h) = smart_size(
            img.width() as usize,
            img.height() as usize,
            unit,
            MIN_IMAGE_TOKENS * area,
            self.max_image_tokens * area,
        );
        let resized = image::imageops::resize(img, w as u32, h as u32, image::imageops::FilterType::Triangle);
        let plane = w * h;
        let mut px = vec![0.0f32; 3 * plane];
        for (i, p) in resized.pixels().enumerate() {
            for ch in 0..3 {
                px[ch * plane + i] = (f32::from(p[ch]) / 255.0 - c.image_mean[ch]) / c.image_std[ch];
            }
        }
        (px, w, h)
    }

    /// Merged token grid `(columns, rows)` for an image of `width` x `height` px.
    pub fn token_grid(&self, width: usize, height: usize) -> (usize, usize) {
        let unit = self.cfg.patch_size * self.cfg.merge;
        (width / unit, height / unit)
    }

    /// Encode planar RGB `[3][height][width]` (already normalized) into
    /// `[tokens][proj_dim]` LLM embeddings, tokens in row-major merged-grid
    /// order. Both sides must be multiples of `patch_size * merge`.
    pub fn encode(&self, pixels: &[f32], width: usize, height: usize) -> Result<Vec<f32>, AllpacaError> {
        self.check_image(pixels, width, height)?;
        let c = &self.cfg;
        let (px, py) = (width / c.patch_size, height / c.patch_size);
        let order = merged_patch_order(px, py);
        let (n, e) = (order.len(), c.n_embd);
        let patches = self.im2col(pixels, width, &order);
        let mut x = linear(&patches, &self.patch_w, &self.patch_b, n, e, patches.len() / n)?;
        add_into(&mut x, &self.position_embedding(px, py, &order));
        let mut normed = vec![0.0f32; n * e];
        for b in &self.blocks {
            layer_norm_rows(&x, &b.ln1_w, &b.ln1_b, &mut normed, e, c.eps);
            let attn = self.attention(b, &normed, &order)?;
            add_into(&mut x, &attn);
            layer_norm_rows(&x, &b.ln2_w, &b.ln2_b, &mut normed, e, c.eps);
            let mut up = linear_f16(&normed, &b.up_w, &b.up_b, n, c.n_ff, e)?;
            up.iter_mut().for_each(|v| *v = gelu_tanh(*v));
            add_into(&mut x, &linear_f16(&up, &b.down_w, &b.down_b, n, e, c.n_ff)?);
        }
        layer_norm_rows(&x, &self.post_w, &self.post_b, &mut normed, e, c.eps);
        // Each run of 4 patches is one 2x2 group, so concatenating them is a
        // reinterpretation as [n / 4][4 * n_embd].
        let tokens = n / 4;
        let mut h = linear_f16(&normed, &self.mm0_w, &self.mm0_b, tokens, 4 * e, 4 * e)?;
        h.iter_mut().for_each(|v| *v = gelu_tanh(*v));
        linear_f16(&h, &self.mm2_w, &self.mm2_b, tokens, c.proj_dim, 4 * e)
    }

    fn check_image(&self, pixels: &[f32], width: usize, height: usize) -> Result<(), AllpacaError> {
        let unit = self.cfg.patch_size * self.cfg.merge;
        if width == 0 || height == 0 || width % unit != 0 || height % unit != 0 {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "qwen3vl image {width}x{height} is not a multiple of {unit}"
            ))));
        }
        if pixels.len() != 3 * width * height {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "qwen3vl pixels are not [3][height][width]".into(),
            )));
        }
        Ok(())
    }

    /// [`Self::encode`] on a HIP device; same inputs and output.
    pub fn encode_on(
        &self,
        hip: &allpaca_hip::HipWorkspace,
        pixels: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, AllpacaError> {
        self.check_image(pixels, width, height)?;
        let c = &self.cfg;
        let (px, py) = (width / c.patch_size, height / c.patch_size);
        let order = merged_patch_order(px, py);
        let (n, e, ff, hd) = (order.len(), c.n_embd, c.n_ff, c.head_dim());
        let (tokens, pp) = (n / 4, 3 * c.patch_size * c.patch_size);
        let patches = self.im2col(pixels, width, &order);
        let pos = self.position_embedding(px, py, &order);
        let rows: Vec<i32> = order.iter().map(|&(y, _)| y as i32).collect();
        let cols: Vec<i32> = order.iter().map(|&(_, x)| x as i32).collect();

        // One transient upload: inputs + every weight, 256-byte aligned parts.
        let mut staged = Staged::default();
        let in_patches = staged.f32(&patches);
        let in_pos = staged.f32(&pos);
        let in_rows = staged.bytes(bytemuck_i32(&rows));
        let in_cols = staged.bytes(bytemuck_i32(&cols));
        let patch_w = staged.f32(&self.patch_w);
        let patch_b = staged.f32(&self.patch_b);
        let blocks: Vec<[usize; 12]> = self
            .blocks
            .iter()
            .map(|b| {
                [
                    staged.f32(&b.ln1_w),
                    staged.f32(&b.ln1_b),
                    staged.f16(&b.qkv_w),
                    staged.f32(&b.qkv_b),
                    staged.f16(&b.out_w),
                    staged.f32(&b.out_b),
                    staged.f32(&b.ln2_w),
                    staged.f32(&b.ln2_b),
                    staged.f16(&b.up_w),
                    staged.f32(&b.up_b),
                    staged.f16(&b.down_w),
                    staged.f32(&b.down_b),
                ]
            })
            .collect();
        let post = [staged.f32(&self.post_w), staged.f32(&self.post_b)];
        let mm = [
            staged.f16(&self.mm0_w),
            staged.f32(&self.mm0_b),
            staged.f16(&self.mm2_w),
            staged.f32(&self.mm2_b),
        ];
        let dev = hip.device;
        let wts = allpaca_hip::DeviceWeights::upload_bytes(dev, &staged.views())?;
        let f = |off: usize| wts.ptr_at(off) as *mut f32;
        let h = |off: usize| wts.ptr_at(off) as *const u16;

        // Activations.
        let acts = allpaca_hip::DeviceWeights::alloc(dev, 4 * (7 * n * e + n * ff + tokens * c.proj_dim))?;
        let act = |i: usize| unsafe { (acts.ptr_at(0) as *mut f32).add(i) };
        let (x, normed, q, k, v, attn, proj) = (act(0), act(n * e), act(2 * n * e), act(3 * n * e), act(4 * n * e), act(5 * n * e), act(6 * n * e));
        let up = act(7 * n * e);
        let out = act(7 * n * e + n * ff);
        let (ni, ei, ffi) = (n as i32, e as i32, ff as i32);

        hip.gemm_nt_dev(f(in_patches), f(patch_w), x, ni, ei, pp as i32)?;
        hip.bias_act_rows_dev(x, f(patch_b), ni, ei, false)?;
        hip.residual_add(x, f(in_pos), ni * ei)?;
        let (prow, pcol) = (wts.ptr_at(in_rows) as *const i32, wts.ptr_at(in_cols) as *const i32);
        for bw in &blocks {
            let [ln1_w, ln1_b, qkv_w, qkv_b, out_w, out_b, ln2_w, ln2_b, up_w, up_b, down_w, down_b] = *bw;
            hip.layernorm_rows_dev(x, f(ln1_w), f(ln1_b), normed, ni, ei, c.eps)?;
            for (part, dst) in [q, k, v].into_iter().enumerate() {
                let w_part = unsafe { h(qkv_w).add(part * e * e) };
                hip.gemm_nt_f16_dev(normed, w_part, dst, ni, ei, ei)?;
                hip.bias_act_rows_dev(dst, unsafe { f(qkv_b).add(part * e) }, ni, ei, false)?;
            }
            for dst in [q, k] {
                hip.vision_rope_rows_dev(dst, prow, pcol, ni, c.n_head as i32, hd as i32, ROPE_THETA)?;
            }
            hip.attention_device(q, k, v, attn, ni, ni, c.n_head as i32, c.n_head as i32, hd as i32, false)?;
            hip.gemm_nt_f16_dev(attn, h(out_w), proj, ni, ei, ei)?;
            hip.bias_act_rows_dev(proj, f(out_b), ni, ei, false)?;
            hip.residual_add(x, proj, ni * ei)?;
            hip.layernorm_rows_dev(x, f(ln2_w), f(ln2_b), normed, ni, ei, c.eps)?;
            hip.gemm_nt_f16_dev(normed, h(up_w), up, ni, ffi, ei)?;
            hip.bias_act_rows_dev(up, f(up_b), ni, ffi, true)?;
            hip.gemm_nt_f16_dev(up, h(down_w), proj, ni, ei, ffi)?;
            hip.bias_act_rows_dev(proj, f(down_b), ni, ei, false)?;
            hip.residual_add(x, proj, ni * ei)?;
        }
        hip.layernorm_rows_dev(x, f(post[0]), f(post[1]), normed, ni, ei, c.eps)?;
        // [n][e] rows read as [n / 4][4e]; the merger's hidden reuses `up`.
        let (ti, e4, pd) = (tokens as i32, 4 * ei, c.proj_dim as i32);
        hip.gemm_nt_f16_dev(normed, h(mm[0]), up, ti, e4, e4)?;
        hip.bias_act_rows_dev(up, f(mm[1]), ti, e4, true)?;
        hip.gemm_nt_f16_dev(up, h(mm[2]), out, ti, pd, e4)?;
        hip.bias_act_rows_dev(out, f(mm[3]), ti, pd, false)?;
        let mut result = vec![0.0f32; tokens * c.proj_dim];
        hip.copy_d2h_f32(out, &mut result)?;
        Ok(result)
    }

    /// Workspace on the configured vision device, or the HIP GPU with the
    /// most free memory right now (`None` without a usable GPU).
    fn pick_hip(&self) -> Option<std::sync::Arc<allpaca_hip::HipWorkspace>> {
        if !allpaca_hip::runtime_linked() || allpaca_hip::device_count() == 0 {
            return None;
        }
        let dev = self.hip_device.or_else(|| {
            (0..allpaca_hip::device_count() as i32)
                .filter_map(|d| allpaca_hip::device_memory_info(d).ok().map(|(free, _)| (d, free)))
                .max_by_key(|&(_, free)| free)
                .map(|(d, _)| d)
        })?;
        let mut cache = self.hip.lock().ok()?;
        if let Some((_, ws)) = cache.iter().find(|(d, _)| *d == dev) {
            return Some(ws.clone());
        }
        let ws = std::sync::Arc::new(allpaca_hip::HipWorkspace::new(dev, 8 << 20, 64).ok()?);
        cache.push((dev, ws.clone()));
        Some(ws)
    }

    /// Run encode on HIP device `device` instead of picking one per call.
    pub fn with_hip_device(mut self, device: Option<i32>) -> Self {
        self.hip_device = device;
        self
    }

    /// Patch pixels `[n][3 * p * p]` (channel, row, column) in `order`.
    fn im2col(&self, pixels: &[f32], width: usize, order: &[(usize, usize)]) -> Vec<f32> {
        let p = self.cfg.patch_size;
        let pp = 3 * p * p;
        let plane = pixels.len() / 3;
        let mut patches = vec![0.0f32; order.len() * pp];
        for (i, &(py, px)) in order.iter().enumerate() {
            let dst = &mut patches[i * pp..(i + 1) * pp];
            for ch in 0..3 {
                for ky in 0..p {
                    let src = ch * plane + (py * p + ky) * width + px * p;
                    dst[(ch * p + ky) * p..(ch * p + ky + 1) * p].copy_from_slice(&pixels[src..src + p]);
                }
            }
        }
        patches
    }

    /// The learned grid, bilinearly resized (align corners) to `px` x `py`, in `order`.
    fn position_embedding(&self, px: usize, py: usize, order: &[(usize, usize)]) -> Vec<f32> {
        let mut x = vec![0.0f32; order.len() * self.cfg.n_embd];
        let (side, e) = (self.cfg.pos_side, self.cfg.n_embd);
        let scale = |out: usize| {
            if out > 1 {
                (side - 1) as f32 / (out - 1) as f32
            } else {
                0.0
            }
        };
        let (sy, sx) = (scale(py), scale(px));
        let at = |yy: usize, xx: usize| &self.pos[(yy * side + xx) * e..(yy * side + xx + 1) * e];
        for (i, &(y, xc)) in order.iter().enumerate() {
            let (fy, fx) = (y as f32 * sy, xc as f32 * sx);
            let (y0, x0) = (fy.floor() as usize, fx.floor() as usize);
            let (y1, x1) = ((y0 + 1).min(side - 1), (x0 + 1).min(side - 1));
            let (wy, wx) = (fy - y0 as f32, fx - x0 as f32);
            let (a, b, cc, d) = (at(y0, x0), at(y0, x1), at(y1, x0), at(y1, x1));
            for (k, v) in x[i * e..(i + 1) * e].iter_mut().enumerate() {
                let top = a[k] * (1.0 - wx) + b[k] * wx;
                let bot = cc[k] * (1.0 - wx) + d[k] * wx;
                *v = top * (1.0 - wy) + bot * wy;
            }
        }
        x
    }

    fn attention(&self, b: &Block, normed: &[f32], order: &[(usize, usize)]) -> Result<Vec<f32>, AllpacaError> {
        let c = &self.cfg;
        let (n, e, hd, nh) = (order.len(), c.n_embd, c.head_dim(), c.n_head);
        let mut qkv = linear_f16(normed, &b.qkv_w, &b.qkv_b, n, 3 * e, e)?;
        for (i, &(y, x)) in order.iter().enumerate() {
            let row = &mut qkv[i * 3 * e..(i + 1) * 3 * e];
            for h in 0..nh {
                vision_rope(&mut row[h * hd..(h + 1) * hd], y, x);
                vision_rope(&mut row[e + h * hd..e + (h + 1) * hd], y, x);
            }
        }
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0.0f32; n * e];
        let mut scores = vec![0.0f32; n];
        let head = |row: usize, part: usize, h: usize| {
            let s = row * 3 * e + part * e + h * hd;
            s..s + hd
        };
        for h in 0..nh {
            for i in 0..n {
                let q = &qkv[head(i, 0, h)];
                for (j, s) in scores.iter_mut().enumerate() {
                    *s = q.iter().zip(&qkv[head(j, 1, h)]).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                allpaca_cpu::softmax(&mut scores);
                let o = &mut out[i * e + h * hd..i * e + (h + 1) * hd];
                for (j, &p) in scores.iter().enumerate() {
                    for (ov, vv) in o.iter_mut().zip(&qkv[head(j, 2, h)]) {
                        *ov += p * vv;
                    }
                }
            }
        }
        linear_f16(&out, &b.out_w, &b.out_b, n, e, e)
    }
}

impl crate::EncoderPlacement for Qwen3VlVision {
    fn device(&self) -> Option<i32> {
        None
    }
}

impl crate::VisionEncoder for Qwen3VlVision {
    fn image_marker(&self) -> Option<&'static str> {
        Some("<|vision_start|><|image_pad|><|vision_end|>")
    }

    fn encode(&self, images: &[crate::ImageInput]) -> Result<crate::VisionTokens, AllpacaError> {
        let mut out = crate::VisionTokens::empty(None);
        for input in images {
            let raw = crate::vision::decode_rgb8(input)?;
            let rgb = image::RgbImage::from_raw(raw.width, raw.height, raw.pixels).ok_or_else(|| {
                AllpacaError::Fatal(FatalError::InvalidConfig("decoded image size mismatch".into()))
            })?;
            let (px, w, h) = self.preprocess(&rgb);
            let embeds = match self.pick_hip() {
                Some(hip) => self.encode_on(&hip, &px, w, h)?,
                None => Qwen3VlVision::encode(self, &px, w, h)?,
            };
            let grid = self.token_grid(w, h);
            out.n_tokens += grid.0 * grid.1;
            out.embeddings.extend_from_slice(&embeds);
            out.grids.push(grid);
        }
        Ok(out)
    }
}

/// llama.cpp `calc_size_preserved_ratio` (transformers `smart_resize`): the
/// nearest `align` multiples, scaled down to `max_pixels` or up to `min_pixels`.
fn smart_size(width: usize, height: usize, align: usize, min_pixels: usize, max_pixels: usize) -> (usize, usize) {
    let f = align as f32;
    let (wf, hf) = (width as f32, height as f32);
    let round = |x: f32| (x / f).round() as usize * align;
    let ceil = |x: f32| (x / f).ceil() as usize * align;
    let floor = |x: f32| (x / f).floor() as usize * align;
    let (mut w, mut h) = (round(wf).max(align), round(hf).max(align));
    if w * h > max_pixels {
        let beta = (hf * wf / max_pixels as f32).sqrt();
        h = floor(hf / beta).max(align);
        w = floor(wf / beta).max(align);
    } else if w * h < min_pixels {
        let beta = (min_pixels as f32 / (hf * wf)).sqrt();
        h = ceil(hf * beta);
        w = ceil(wf * beta);
    }
    (w, h)
}

/// Patches `(row, column)` in llama.cpp's 2x2 spatial-merge order: each merged
/// cell's four patches are consecutive, cells in row-major order.
fn merged_patch_order(px: usize, py: usize) -> Vec<(usize, usize)> {
    let mut order = Vec::with_capacity(px * py);
    for y in (0..py).step_by(2) {
        for x in (0..px).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    order.push((y + dy, x + dx));
                }
            }
        }
    }
    order
}

/// ggml `GGML_ROPE_TYPE_VISION` with sections `[d/4; 4]`: pairs `(i, i + d/2)`;
/// the first quarter of pairs turns with the patch row and the second with its
/// column, each quarter restarting the frequency ladder.
fn vision_rope(head: &mut [f32], row: usize, col: usize) {
    let half = head.len() / 2;
    let quarter = half / 2;
    let theta_scale = ROPE_THETA.powf(-2.0 / half as f32);
    for i in 0..half {
        let (pos, k) = if i < quarter { (row, i) } else { (col, i - quarter) };
        let theta = pos as f32 * theta_scale.powi(k as i32);
        let (sin, cos) = theta.sin_cos();
        let (x0, x1) = (head[i], head[i + half]);
        head[i] = x0 * cos - x1 * sin;
        head[i + half] = x0 * sin + x1 * cos;
    }
}

fn layer_norm_rows(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32], n: usize, eps: f32) {
    for (src, dst) in x.chunks(n).zip(out.chunks_mut(n)) {
        let mean = src.iter().sum::<f32>() / n as f32;
        let var = src.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..n {
            dst[i] = (src[i] - mean) * inv * w[i] + b[i];
        }
    }
}

/// Host staging for one contiguous device upload; parts start 256-byte aligned.
#[derive(Default)]
struct Staged {
    buf: Vec<u8>,
}

impl Staged {
    fn bytes(&mut self, b: &[u8]) -> usize {
        let off = self.buf.len().div_ceil(256) * 256;
        self.buf.resize(off, 0);
        self.buf.extend_from_slice(b);
        off
    }

    fn f32(&mut self, v: &[f32]) -> usize {
        let off = self.bytes(&[]);
        self.buf.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        off
    }

    fn f16(&mut self, v: &[u16]) -> usize {
        let off = self.bytes(&[]);
        self.buf.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        off
    }

    fn views(&self) -> Vec<&[u8]> {
        vec![&self.buf]
    }
}

fn bytemuck_i32(v: &[i32]) -> &[u8] {
    // SAFETY: i32 has no padding and any byte pattern of u8 is valid.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// [`linear`] with F16 weights (converted for the call).
fn linear_f16(x: &[f32], w: &[u16], bias: &[f32], rows: usize, n: usize, k: usize) -> Result<Vec<f32>, AllpacaError> {
    let w: Vec<f32> = w.iter().map(|&b| allpaca_cpu::f16_to_f32(b)).collect();
    linear(x, &w, bias, rows, n, k)
}

/// `y[rows][n] = x[rows][k] * w[n][k]^T + bias`.
fn linear(x: &[f32], w: &[f32], bias: &[f32], rows: usize, n: usize, k: usize) -> Result<Vec<f32>, AllpacaError> {
    let mut y = vec![0.0f32; rows * n];
    allpaca_cpu::matmul_nt(x, w, &mut y, rows, n, k)?;
    for row in y.chunks_mut(n) {
        for (v, b) in row.iter_mut().zip(bias) {
            *v += b;
        }
    }
    Ok(y)
}

fn add_into(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// GELU, tanh approximation (ggml `ggml_gelu`, HF `gelu_pytorch_tanh`).
fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mmproj_path() -> Option<std::path::PathBuf> {
        let p = std::env::var_os("ALLPACA_MMPROJ_GGUF")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/Qwen3.8-27B-GGUF/mmproj-F16.gguf")
            });
        p.exists().then_some(p)
    }

    /// llama.cpp `llama-mtmd-debug --image rainbow`: raw [0, 1] RGB, planar.
    fn rainbow_chw(size: usize) -> Vec<f32> {
        let n = size * size;
        let mut img = vec![0.0f32; 3 * n];
        let c = size as f32 / 2.0;
        let max_dist = (c * c + c * c).sqrt();
        for y in 0..size {
            for x in 0..size {
                let (dx, dy) = (x as f32 - c, y as f32 - c);
                let mut hue = dy.atan2(dx) / (2.0 * 3.141_592_65_f32);
                if hue < 0.0 {
                    hue += 1.0;
                }
                let sat = ((dx * dx + dy * dy).sqrt() / max_dist).min(1.0);
                let h6 = hue * 6.0;
                let i6 = h6 as i32;
                let f = h6 - i6 as f32;
                let (p, q, t) = (1.0 - sat, 1.0 - sat * f, 1.0 - sat * (1.0 - f));
                let (r, g, b) = match i6 % 6 {
                    0 => (1.0, t, p),
                    1 => (q, 1.0, p),
                    2 => (p, 1.0, t),
                    3 => (p, q, 1.0),
                    4 => (t, p, 1.0),
                    _ => (1.0, p, q),
                };
                img[y * size + x] = r;
                img[n + y * size + x] = g;
                img[2 * n + y * size + x] = b;
            }
        }
        img
    }

    #[test]
    fn smart_size_matches_llama_cpp_dyn_size_preprocessor() {
        // align 32, 8..=1024 tokens of 32x32 px (llama.cpp calc_size_preserved_ratio).
        let (min_px, max_px) = (8 * 1024, 1024 * 1024);
        assert_eq!(smart_size(1000, 600, 32, min_px, max_px), (992, 608));
        assert_eq!(smart_size(4000, 3000, 32, min_px, max_px), (1152, 864));
        assert_eq!(smart_size(10, 10, 32, min_px, max_px), (96, 96));
    }

    #[test]
    fn qwen3vl_encoder_reports_each_image_grid() {
        let Some(path) = mmproj_path() else {
            eprintln!("skip: no Qwen3.8 mmproj GGUF");
            return;
        };
        let vision = Qwen3VlVision::load_gguf(&path).unwrap();
        let img = image::RgbImage::from_fn(64, 64, |x, y| image::Rgb([x as u8 * 4, y as u8 * 4, 128]));
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let out = crate::VisionEncoder::encode(&vision, &[crate::ImageInput::new(png, "image/png")]).unwrap();
        // 64x64 is below the 8-token minimum, so it is upscaled to 96x96: 3x3 tokens.
        assert_eq!(out.grids, vec![(3, 3)]);
        assert_eq!(out.n_tokens, 9);
        assert_eq!(out.embeddings.len(), 9 * 5120);
    }

    #[cfg(feature = "hip")]
    #[test]
    fn qwen3vl_hip_encode_matches_cpu() {
        let Some(path) = mmproj_path() else {
            eprintln!("skip: no Qwen3.8 mmproj GGUF");
            return;
        };
        if allpaca_hip::device_count() == 0 {
            eprintln!("skip: no HIP device");
            return;
        }
        let vision = Qwen3VlVision::load_gguf(&path).unwrap();
        // Non-square 96x64 px: 6x4 patches, 3x2 merged tokens.
        let (w, h) = (96usize, 64usize);
        let px: Vec<f32> = (0..3 * w * h).map(|i| ((i * 37) % 101) as f32 / 50.0 - 1.0).collect();
        let cpu = vision.encode(&px, w, h).unwrap();
        let hip = allpaca_hip::HipWorkspace::new(0, 8 << 20, 64).unwrap();
        let gpu = vision.encode_on(&hip, &px, w, h).unwrap();
        assert_eq!(gpu.len(), 6 * 5120);
        let scale = cpu.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = cpu.iter().zip(&gpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 2e-3 * scale.max(1.0), "max |cpu - hip| {worst} (scale {scale})");
    }

    #[test]
    fn qwen3vl_mmproj_encodes_like_llama_cpp() {
        let Some(path) = mmproj_path() else {
            eprintln!("skip: no Qwen3.8 mmproj GGUF");
            return;
        };
        let vision = Qwen3VlVision::load_gguf(&path).unwrap();
        assert_eq!(vision.config().proj_dim, 5120);
        let out = vision.encode(&rainbow_chw(64), 64, 64).unwrap();
        // 64x64 = 4x4 patches = 2x2 merged tokens.
        assert_eq!(out.len(), 4 * 5120);
        // Reference: `llama-mtmd-debug -p encode -n 64 --image rainbow` on the
        // same mmproj (projector output, printed to 4 decimals). ggml's CPU
        // path rounds activations to F16 before F16 matmuls; this F32 encoder
        // lands within ~1.3e-3 per value and 0.13 on the sum.
        let want = [
            ([0.2987, 0.1001, -0.1338], [0.8652, -0.7776, 0.3556]),
            ([0.1703, -0.0759, -0.4602], [-0.3813, -0.7509, -0.0301]),
            ([0.2119, 0.2596, -0.2687], [-0.3585, -0.4774, 0.6375]),
            ([0.1490, -0.0222, -0.1097], [-0.1295, -0.0441, 0.2095]),
        ];
        for (r, (head, tail)) in want.iter().enumerate() {
            let row = &out[r * 5120..(r + 1) * 5120];
            let (got_head, got_tail) = (&row[..3], &row[5117..]);
            for (g, w) in got_head.iter().chain(got_tail).zip(head.iter().chain(tail)) {
                assert!((g - w).abs() < 5e-3, "row {r}: got {got_head:?}..{got_tail:?}");
            }
        }
        let sum: f64 = out.iter().map(|&v| f64::from(v)).sum();
        assert!((sum + 345.888824).abs() < 0.5, "sum {sum}, llama.cpp -345.888824");
    }
}
