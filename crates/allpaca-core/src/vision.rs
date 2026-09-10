//! CPU patch-embed vision encoder with optional ViT blocks and loadable F32 weights.
//!
//! Decodes JPEG/PNG/WebP via `image`. AVIF (and other formats `image` rejects)
//! fall back to an `ffmpeg` pipe → PNG when `ffmpeg` is on PATH.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;

use allpaca_safetensors::{write_safetensors, SafetensorsDtype, SafetensorsFile, SafetensorsWriteTensor};
use allpaca_types::{AllpacaError, FatalError};

use crate::modality::{EncoderPlacement, ImageInput, VisionEncoder, VisionTokens};

/// Default square resize before patching.
pub const DEFAULT_IMAGE_SIZE: u32 = 64;
/// Default patch edge in pixels.
pub const DEFAULT_PATCH_SIZE: u32 = 16;

const LN_EPS: f32 = 1e-5;

/// One ViT transformer block (pre-norm: LN → attn → residual → LN → MLP → residual).
#[derive(Debug, Clone)]
pub struct VitBlockWeights {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub attn_qkv_w: Vec<f32>,
    pub attn_qkv_b: Vec<f32>,
    pub attn_proj_w: Vec<f32>,
    pub attn_proj_b: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub mlp_fc1_w: Vec<f32>,
    pub mlp_fc1_b: Vec<f32>,
    pub mlp_fc2_w: Vec<f32>,
    pub mlp_fc2_b: Vec<f32>,
}

/// Linear patch-embed → optional ViT blocks → soft tokens for LM prefill.
pub struct PatchVisionEncoder {
    device: Option<i32>,
    out_dim: usize,
    image_size: u32,
    patch_size: u32,
    n_heads: usize,
    /// Row-major `[out_dim, patch_dim]` where `patch_dim = patch*patch*3`.
    weight: Arc<Vec<f32>>,
    bias: Arc<Vec<f32>>,
    blocks: Arc<Vec<VitBlockWeights>>,
}

impl PatchVisionEncoder {
    pub fn new(
        device: Option<i32>,
        out_dim: usize,
        image_size: u32,
        patch_size: u32,
        weight: Vec<f32>,
        bias: Vec<f32>,
        blocks: Vec<VitBlockWeights>,
    ) -> Result<Self, AllpacaError> {
        if image_size == 0 || patch_size == 0 || image_size % patch_size != 0 {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "vision image_size {image_size} must be a positive multiple of patch_size {patch_size}"
            ))));
        }
        let patch_dim = (patch_size as usize) * (patch_size as usize) * 3;
        let need = out_dim.checked_mul(patch_dim).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::InvalidConfig(
                "vision weight size overflow".into(),
            ))
        })?;
        if weight.len() != need {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "vision weight len {} != out_dim {out_dim} * patch_dim {patch_dim}",
                weight.len()
            ))));
        }
        if bias.len() != out_dim {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "vision bias len {} != out_dim {out_dim}",
                bias.len()
            ))));
        }
        let n_heads = pick_n_heads(out_dim);
        for (i, b) in blocks.iter().enumerate() {
            validate_block(b, out_dim, i)?;
        }
        Ok(Self {
            device,
            out_dim,
            image_size,
            patch_size,
            n_heads,
            weight: Arc::new(weight),
            bias: Arc::new(bias),
            blocks: Arc::new(blocks),
        })
    }

    /// Deterministic synthetic weights for tests / CPU-only synthetic Engine.
    pub fn synthetic(device: Option<i32>, out_dim: usize, seed: u64) -> Result<Self, AllpacaError> {
        Self::synthetic_sized(
            device,
            out_dim,
            DEFAULT_IMAGE_SIZE,
            DEFAULT_PATCH_SIZE,
            seed,
            0,
        )
    }

    pub fn synthetic_sized(
        device: Option<i32>,
        out_dim: usize,
        image_size: u32,
        patch_size: u32,
        seed: u64,
        n_blocks: u32,
    ) -> Result<Self, AllpacaError> {
        let patch_dim = (patch_size as usize) * (patch_size as usize) * 3;
        let mut state = seed ^ 0xC0FFEE_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            ((state >> 33) as f32) * (1.0 / (u32::MAX as f32)) - 0.5
        };
        let mut weight = vec![0.0f32; out_dim * patch_dim];
        let mut bias = vec![0.0f32; out_dim];
        for w in weight.iter_mut() {
            *w = next() * 0.02;
        }
        for b in bias.iter_mut() {
            *b = next() * 0.01;
        }
        let mut blocks = Vec::with_capacity(n_blocks as usize);
        for _ in 0..n_blocks {
            blocks.push(synthetic_block(out_dim, &mut next)?);
        }
        Self::new(device, out_dim, image_size, patch_size, weight, bias, blocks)
    }

    pub fn load(
        path: &Path,
        device: Option<i32>,
        out_dim: usize,
        image_size: u32,
        patch_size: u32,
        n_blocks: u32,
    ) -> Result<Self, AllpacaError> {
        let file = SafetensorsFile::open(path)?;
        let weight = file.tensor_f32("patch_proj.weight")?;
        let bias = match file.tensor_f32("patch_proj.bias") {
            Ok(b) => b,
            Err(_) => vec![0.0f32; out_dim],
        };
        let mut blocks = Vec::with_capacity(n_blocks as usize);
        for i in 0..n_blocks {
            blocks.push(load_block(&file, out_dim, i as usize)?);
        }
        Self::new(device, out_dim, image_size, patch_size, weight, bias, blocks)
    }

    pub fn write_safetensors(&self, path: &Path) -> Result<(), AllpacaError> {
        let patch_dim = (self.patch_size as usize) * (self.patch_size as usize) * 3;
        let mlp_dim = mlp_hidden(self.out_dim);
        let mut tensors = vec![
            SafetensorsWriteTensor {
                name: "patch_proj.weight".into(),
                dims: vec![self.out_dim as u64, patch_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: self.weight.as_slice(),
            },
            SafetensorsWriteTensor {
                name: "patch_proj.bias".into(),
                dims: vec![self.out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: self.bias.as_slice(),
            },
        ];
        for (i, b) in self.blocks.iter().enumerate() {
            let p = format!("blocks.{i}");
            tensors.push(SafetensorsWriteTensor {
                name: format!("{p}.ln1.weight"),
                dims: vec![self.out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &b.ln1_w,
            });
            tensors.push(SafetensorsWriteTensor {
                name: format!("{p}.ln1.bias"),
                dims: vec![self.out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &b.ln1_b,
            });
            tensors.push(SafetensorsWriteTensor {
                name: format!("{p}.attn_qkv.weight"),
                dims: vec![(3 * self.out_dim) as u64, self.out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &b.attn_qkv_w,
            });
            tensors.push(SafetensorsWriteTensor {
                name: format!("{p}.attn_qkv.bias"),
                dims: vec![(3 * self.out_dim) as u64],
                dtype: SafetensorsDtype::F32,
                data: &b.attn_qkv_b,
            });
            tensors.push(SafetensorsWriteTensor {
                name: format!("{p}.attn_proj.weight"),
                dims: vec![self.out_dim as u64, self.out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &b.attn_proj_w,
            });
            tensors.push(SafetensorsWriteTensor {
                name: format!("{p}.attn_proj.bias"),
                dims: vec![self.out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &b.attn_proj_b,
            });
            tensors.push(SafetensorsWriteTensor {
                name: format!("{p}.ln2.weight"),
                dims: vec![self.out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &b.ln2_w,
            });
            tensors.push(SafetensorsWriteTensor {
                name: format!("{p}.ln2.bias"),
                dims: vec![self.out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &b.ln2_b,
            });
            tensors.push(SafetensorsWriteTensor {
                name: format!("{p}.mlp.fc1.weight"),
                dims: vec![mlp_dim as u64, self.out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &b.mlp_fc1_w,
            });
            tensors.push(SafetensorsWriteTensor {
                name: format!("{p}.mlp.fc1.bias"),
                dims: vec![mlp_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &b.mlp_fc1_b,
            });
            tensors.push(SafetensorsWriteTensor {
                name: format!("{p}.mlp.fc2.weight"),
                dims: vec![self.out_dim as u64, mlp_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &b.mlp_fc2_w,
            });
            tensors.push(SafetensorsWriteTensor {
                name: format!("{p}.mlp.fc2.bias"),
                dims: vec![self.out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &b.mlp_fc2_b,
            });
        }
        write_safetensors(path, &tensors)
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    pub fn n_patches(&self) -> usize {
        let n = (self.image_size / self.patch_size) as usize;
        n * n
    }

    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }
}

impl EncoderPlacement for PatchVisionEncoder {
    fn device(&self) -> Option<i32> {
        self.device
    }
}

impl VisionEncoder for PatchVisionEncoder {
    fn encode(&self, images: &[ImageInput]) -> Result<VisionTokens, AllpacaError> {
        if images.is_empty() {
            return Ok(VisionTokens::empty(self.device));
        }
        let mut embeddings = Vec::new();
        let mut n_tokens = 0usize;
        for img in images {
            let rgb = decode_rgb8(img)?;
            let resized = resize_rgb8(&rgb, self.image_size, self.image_size)?;
            let mut tokens = patch_embed(
                &resized,
                self.image_size,
                self.patch_size,
                self.out_dim,
                self.weight.as_slice(),
                self.bias.as_slice(),
            )?;
            for block in self.blocks.iter() {
                apply_vit_block(&mut tokens, self.out_dim, self.n_heads, block)?;
            }
            n_tokens += tokens.len() / self.out_dim;
            embeddings.extend(tokens);
        }
        Ok(VisionTokens {
            embeddings,
            n_tokens,
            device: self.device,
        })
    }
}

fn pick_n_heads(dim: usize) -> usize {
    for h in [8usize, 4, 2, 1] {
        if dim % h == 0 {
            return h;
        }
    }
    1
}

fn mlp_hidden(dim: usize) -> usize {
    dim.saturating_mul(4).max(1)
}

fn validate_block(b: &VitBlockWeights, dim: usize, i: usize) -> Result<(), AllpacaError> {
    let mlp = mlp_hidden(dim);
    let check = |name: &str, got: usize, want: usize| -> Result<(), AllpacaError> {
        if got != want {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "vision blocks.{i}.{name} len {got} != {want}"
            ))));
        }
        Ok(())
    };
    check("ln1.weight", b.ln1_w.len(), dim)?;
    check("ln1.bias", b.ln1_b.len(), dim)?;
    check("attn_qkv.weight", b.attn_qkv_w.len(), 3 * dim * dim)?;
    check("attn_qkv.bias", b.attn_qkv_b.len(), 3 * dim)?;
    check("attn_proj.weight", b.attn_proj_w.len(), dim * dim)?;
    check("attn_proj.bias", b.attn_proj_b.len(), dim)?;
    check("ln2.weight", b.ln2_w.len(), dim)?;
    check("ln2.bias", b.ln2_b.len(), dim)?;
    check("mlp.fc1.weight", b.mlp_fc1_w.len(), mlp * dim)?;
    check("mlp.fc1.bias", b.mlp_fc1_b.len(), mlp)?;
    check("mlp.fc2.weight", b.mlp_fc2_w.len(), dim * mlp)?;
    check("mlp.fc2.bias", b.mlp_fc2_b.len(), dim)?;
    Ok(())
}

fn synthetic_block(dim: usize, next: &mut impl FnMut() -> f32) -> Result<VitBlockWeights, AllpacaError> {
    let mlp = mlp_hidden(dim);
    let fill = |n: usize, scale: f32, next: &mut dyn FnMut() -> f32| {
        let mut v = vec![0.0f32; n];
        for x in v.iter_mut() {
            *x = next() * scale;
        }
        v
    };
    Ok(VitBlockWeights {
        ln1_w: fill(dim, 0.02, next),
        ln1_b: fill(dim, 0.01, next),
        attn_qkv_w: fill(3 * dim * dim, 0.02, next),
        attn_qkv_b: fill(3 * dim, 0.01, next),
        attn_proj_w: fill(dim * dim, 0.02, next),
        attn_proj_b: fill(dim, 0.01, next),
        ln2_w: fill(dim, 0.02, next),
        ln2_b: fill(dim, 0.01, next),
        mlp_fc1_w: fill(mlp * dim, 0.02, next),
        mlp_fc1_b: fill(mlp, 0.01, next),
        mlp_fc2_w: fill(dim * mlp, 0.02, next),
        mlp_fc2_b: fill(dim, 0.01, next),
    })
}

fn load_block(
    file: &SafetensorsFile,
    dim: usize,
    i: usize,
) -> Result<VitBlockWeights, AllpacaError> {
    let p = format!("blocks.{i}");
    let or_zeros = |name: &str, n: usize| -> Result<Vec<f32>, AllpacaError> {
        match file.tensor_f32(name) {
            Ok(v) => Ok(v),
            Err(_) => Ok(vec![0.0f32; n]),
        }
    };
    let b = VitBlockWeights {
        ln1_w: or_zeros(&format!("{p}.ln1.weight"), dim)?,
        ln1_b: or_zeros(&format!("{p}.ln1.bias"), dim)?,
        attn_qkv_w: file.tensor_f32(&format!("{p}.attn_qkv.weight"))?,
        attn_qkv_b: or_zeros(&format!("{p}.attn_qkv.bias"), 3 * dim)?,
        attn_proj_w: file.tensor_f32(&format!("{p}.attn_proj.weight"))?,
        attn_proj_b: or_zeros(&format!("{p}.attn_proj.bias"), dim)?,
        ln2_w: or_zeros(&format!("{p}.ln2.weight"), dim)?,
        ln2_b: or_zeros(&format!("{p}.ln2.bias"), dim)?,
        mlp_fc1_w: file.tensor_f32(&format!("{p}.mlp.fc1.weight"))?,
        mlp_fc1_b: or_zeros(&format!("{p}.mlp.fc1.bias"), mlp_hidden(dim))?,
        mlp_fc2_w: file.tensor_f32(&format!("{p}.mlp.fc2.weight"))?,
        mlp_fc2_b: or_zeros(&format!("{p}.mlp.fc2.bias"), dim)?,
    };
    validate_block(&b, dim, i)?;
    Ok(b)
}

fn apply_vit_block(
    tokens: &mut [f32],
    dim: usize,
    n_heads: usize,
    block: &VitBlockWeights,
) -> Result<(), AllpacaError> {
    let seq = tokens.len() / dim;
    if seq * dim != tokens.len() || seq == 0 {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "vision vit block token shape mismatch".into(),
        )));
    }
    let head_dim = dim / n_heads;
    let mut normed = vec![0.0f32; tokens.len()];
    for t in 0..seq {
        let row = &tokens[t * dim..(t + 1) * dim];
        layer_norm(row, &block.ln1_w, &block.ln1_b, &mut normed[t * dim..(t + 1) * dim]);
    }
    let mut qkv = vec![0.0f32; seq * 3 * dim];
    for t in 0..seq {
        linear(
            &normed[t * dim..(t + 1) * dim],
            &block.attn_qkv_w,
            &block.attn_qkv_b,
            3 * dim,
            dim,
            &mut qkv[t * 3 * dim..(t + 1) * 3 * dim],
        );
    }
    let mut attn_out = vec![0.0f32; seq * dim];
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq];
    for h in 0..n_heads {
        for qi in 0..seq {
            let q_base = qi * 3 * dim + h * head_dim;
            for kj in 0..seq {
                let k_base = kj * 3 * dim + dim + h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += qkv[q_base + d] * qkv[k_base + d];
                }
                scores[kj] = dot * scale;
            }
            softmax_inplace(&mut scores);
            let out_base = qi * dim + h * head_dim;
            for d in 0..head_dim {
                attn_out[out_base + d] = 0.0;
            }
            for kj in 0..seq {
                let v_base = kj * 3 * dim + 2 * dim + h * head_dim;
                let a = scores[kj];
                for d in 0..head_dim {
                    attn_out[out_base + d] += a * qkv[v_base + d];
                }
            }
        }
    }
    let mut proj = vec![0.0f32; seq * dim];
    for t in 0..seq {
        linear(
            &attn_out[t * dim..(t + 1) * dim],
            &block.attn_proj_w,
            &block.attn_proj_b,
            dim,
            dim,
            &mut proj[t * dim..(t + 1) * dim],
        );
    }
    for i in 0..tokens.len() {
        tokens[i] += proj[i];
    }
    for t in 0..seq {
        let row = &tokens[t * dim..(t + 1) * dim];
        layer_norm(row, &block.ln2_w, &block.ln2_b, &mut normed[t * dim..(t + 1) * dim]);
    }
    let mlp = mlp_hidden(dim);
    let mut hidden = vec![0.0f32; mlp];
    let mut mlp_out = vec![0.0f32; dim];
    for t in 0..seq {
        linear(
            &normed[t * dim..(t + 1) * dim],
            &block.mlp_fc1_w,
            &block.mlp_fc1_b,
            mlp,
            dim,
            &mut hidden,
        );
        for x in hidden.iter_mut() {
            *x = gelu(*x);
        }
        linear(
            &hidden,
            &block.mlp_fc2_w,
            &block.mlp_fc2_b,
            dim,
            mlp,
            &mut mlp_out,
        );
        let dst = &mut tokens[t * dim..(t + 1) * dim];
        for d in 0..dim {
            dst[d] += mlp_out[d];
        }
    }
    Ok(())
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32]) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let mut var = 0.0f32;
    for &v in x {
        let d = v - mean;
        var += d * d;
    }
    var /= n;
    let inv = 1.0 / (var + LN_EPS).sqrt();
    for i in 0..x.len() {
        out[i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}

fn linear(x: &[f32], weight: &[f32], bias: &[f32], out_dim: usize, in_dim: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(out.len(), out_dim);
    debug_assert_eq!(weight.len(), out_dim * in_dim);
    for o in 0..out_dim {
        let mut acc = bias[o];
        let row = &weight[o * in_dim..(o + 1) * in_dim];
        for k in 0..in_dim {
            acc += row[k] * x[k];
        }
        out[o] = acc;
    }
}

fn gelu(x: f32) -> f32 {
    // tanh approximation
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

fn softmax_inplace(x: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

#[derive(Debug, Clone)]
struct RgbImage {
    width: u32,
    height: u32,
    /// Row-major RGB8.
    pixels: Vec<u8>,
}

fn decode_rgb8(input: &ImageInput) -> Result<RgbImage, AllpacaError> {
    match image::load_from_memory(&input.bytes) {
        Ok(dyn_img) => {
            let rgb = dyn_img.to_rgb8();
            Ok(RgbImage {
                width: rgb.width(),
                height: rgb.height(),
                pixels: rgb.into_raw(),
            })
        }
        Err(e) => {
            // AVIF and other formats: ffmpeg → PNG when available.
            if let Ok(png) = ffmpeg_to_png(&input.bytes) {
                let dyn_img = image::load_from_memory(&png).map_err(|e2| {
                    AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "vision decode after ffmpeg: {e2} (image: {e})"
                    )))
                })?;
                let rgb = dyn_img.to_rgb8();
                return Ok(RgbImage {
                    width: rgb.width(),
                    height: rgb.height(),
                    pixels: rgb.into_raw(),
                });
            }
            Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "vision decode failed for mime {}: {e} (install ffmpeg for AVIF)",
                input.mime
            ))))
        }
    }
}

fn ffmpeg_to_png(bytes: &[u8]) -> Result<Vec<u8>, AllpacaError> {
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            AllpacaError::Fatal(FatalError::Io(format!(
                "spawn ffmpeg for vision decode: {e}"
            )))
        })?;
    {
        let stdin = child.stdin.as_mut().ok_or_else(|| {
            AllpacaError::Fatal(FatalError::Io("ffmpeg stdin missing".into()))
        })?;
        stdin.write_all(bytes).map_err(|e| {
            AllpacaError::Fatal(FatalError::Io(format!("ffmpeg stdin write: {e}")))
        })?;
    }
    let out = child.wait_with_output().map_err(|e| {
        AllpacaError::Fatal(FatalError::Io(format!("ffmpeg wait: {e}")))
    })?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(AllpacaError::Fatal(FatalError::Io(format!(
            "ffmpeg vision decode failed: {err}"
        ))));
    }
    if out.stdout.len() < 8 {
        return Err(AllpacaError::Fatal(FatalError::Io(
            "ffmpeg produced empty PNG".into(),
        )));
    }
    Ok(out.stdout)
}

fn resize_rgb8(img: &RgbImage, tw: u32, th: u32) -> Result<RgbImage, AllpacaError> {
    if img.width == tw && img.height == th {
        return Ok(img.clone());
    }
    let src = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone()).ok_or_else(
        || AllpacaError::Fatal(FatalError::ModelLoad("invalid rgb buffer".into())),
    )?;
    let resized = image::imageops::resize(&src, tw, th, image::imageops::FilterType::Triangle);
    Ok(RgbImage {
        width: tw,
        height: th,
        pixels: resized.into_raw(),
    })
}

fn patch_embed(
    img: &RgbImage,
    image_size: u32,
    patch_size: u32,
    out_dim: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>, AllpacaError> {
    debug_assert_eq!(img.width, image_size);
    debug_assert_eq!(img.height, image_size);
    let n = (image_size / patch_size) as usize;
    let patch_dim = (patch_size as usize) * (patch_size as usize) * 3;
    let mut out = vec![0.0f32; n * n * out_dim];
    let mut patch = vec![0.0f32; patch_dim];
    for py in 0..n {
        for px in 0..n {
            let mut i = 0usize;
            for y in 0..patch_size as usize {
                let row = (py * patch_size as usize + y) * image_size as usize * 3;
                for x in 0..patch_size as usize {
                    let off = row + (px * patch_size as usize + x) * 3;
                    patch[i] = img.pixels[off] as f32 / 255.0;
                    patch[i + 1] = img.pixels[off + 1] as f32 / 255.0;
                    patch[i + 2] = img.pixels[off + 2] as f32 / 255.0;
                    i += 3;
                }
            }
            let token = py * n + px;
            let dst = &mut out[token * out_dim..(token + 1) * out_dim];
            for o in 0..out_dim {
                let mut acc = bias[o];
                let row = &weight[o * patch_dim..(o + 1) * patch_dim];
                for k in 0..patch_dim {
                    acc += row[k] * patch[k];
                }
                dst[o] = acc;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_jpeg() -> Vec<u8> {
        let img = image::RgbImage::from_pixel(8, 8, image::Rgb([200, 20, 20]));
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut buf),
                image::ImageFormat::Jpeg,
            )
            .unwrap();
        buf
    }

    #[test]
    fn synthetic_encoder_roundtrips_jpeg_bytes() {
        let enc = PatchVisionEncoder::synthetic(Some(1), 32, 7).unwrap();
        assert_eq!(enc.n_patches(), 16);
        let tokens = enc
            .encode(&[ImageInput::new(tiny_jpeg(), "image/jpeg")])
            .unwrap();
        assert_eq!(tokens.n_tokens, 16);
        assert_eq!(tokens.embeddings.len(), 16 * 32);
        assert_eq!(tokens.device, Some(1));
        assert!(tokens.embeddings.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn vit_block_encode_runs_and_differs_from_patch_only() {
        let jpeg = tiny_jpeg();
        let patch_only =
            PatchVisionEncoder::synthetic_sized(Some(1), 32, 64, 16, 7, 0).unwrap();
        let vit_lite =
            PatchVisionEncoder::synthetic_sized(Some(1), 32, 64, 16, 7, 1).unwrap();
        assert_eq!(vit_lite.n_blocks(), 1);
        let a = patch_only
            .encode(&[ImageInput::new(jpeg.clone(), "image/jpeg")])
            .unwrap();
        let b = vit_lite
            .encode(&[ImageInput::new(jpeg, "image/jpeg")])
            .unwrap();
        assert_eq!(a.n_tokens, b.n_tokens);
        assert_eq!(b.device, Some(1));
        assert_ne!(a.embeddings, b.embeddings);
        assert!(b.embeddings.iter().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn load_written_weights_matches_synthetic() {
        let enc = PatchVisionEncoder::synthetic(None, 32, 99).unwrap();
        let path = std::env::temp_dir().join("allpaca_vision_patch_test.safetensors");
        enc.write_safetensors(&path).unwrap();
        let loaded =
            PatchVisionEncoder::load(&path, Some(0), 32, DEFAULT_IMAGE_SIZE, DEFAULT_PATCH_SIZE, 0)
                .unwrap();
        let jpeg = {
            let img = image::RgbImage::from_pixel(16, 16, image::Rgb([10, 20, 30]));
            let mut buf = Vec::new();
            image::DynamicImage::ImageRgb8(img)
                .write_to(
                    &mut std::io::Cursor::new(&mut buf),
                    image::ImageFormat::Jpeg,
                )
                .unwrap();
            buf
        };
        let a = enc.encode(&[ImageInput::new(jpeg.clone(), "image/jpeg")]).unwrap();
        let b = loaded
            .encode(&[ImageInput::new(jpeg, "image/jpeg")])
            .unwrap();
        assert_eq!(a.embeddings, b.embeddings);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn write_repo_dim32_vision_weights_fixture() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/vision_patch_dim32.safetensors");
        let enc = PatchVisionEncoder::synthetic(None, 32, 42).unwrap();
        enc.write_safetensors(&path).unwrap();
        assert!(path.is_file());
    }

    #[test]
    fn encodes_https_avif_yeelight_via_ffmpeg() {
        let url = "https://spyn.dev/_astro/yeelight.B4dLTFRe_2tsYNB.avif";
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap();
        let bytes = client.get(url).send().unwrap().bytes().unwrap().to_vec();
        assert!(bytes.len() > 10_000);
        let enc = PatchVisionEncoder::synthetic(Some(1), 32, 42).unwrap();
        let tokens = enc
            .encode(&[ImageInput::new(bytes, "image/avif")])
            .expect("AVIF decode needs ffmpeg on PATH");
        assert_eq!(tokens.n_tokens, enc.n_patches());
        assert_eq!(tokens.embeddings.len(), tokens.n_tokens * 32);
        assert!(tokens.embeddings.iter().any(|v| v.abs() > 1e-6));
    }
}
