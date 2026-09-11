//! CPU/HIP patch-embed vision encoder with optional ViT blocks and loadable F32 weights.
//!
//! HIP residency uploads patch + block tensors once; encode prefers device
//! `gemm_nt_dev_b` for patch linear and ViT block linears, and
//! `HipWorkspace::attention` (online softmax) for ViT self-attn. LayerNorm
//! stays host: no mean/var LN kernel (RMSNorm is not equivalent). SafeTensors
//! load accepts native `patch_proj` / `blocks.*` and CLIP-style `visual.proj` /
//! `visual.transformer.resblocks.*` aliases.
//!
//! Decodes JPEG/PNG/WebP via `image`. AVIF (and other formats `image` rejects)
//! fall back to an `ffmpeg` pipe → PNG when `ffmpeg` is on PATH.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(test)]
static HIP_VIT_ATTN_HITS: AtomicUsize = AtomicUsize::new(0);

use allpaca_hip::DeviceWeights;
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
    /// Device-resident pack of `weight` + `bias` + ViT block tensors (lifetime of encoder).
    resident: Mutex<Option<DeviceWeights>>,
    /// CLIP `visual.class_embedding` (prepended as token 0).
    class_emb: Option<Vec<f32>>,
    /// CLIP `visual.positional_embedding` `[seq, dim]` including class token when present.
    pos_emb: Option<Vec<f32>>,
    ln_pre_w: Option<Vec<f32>>,
    ln_pre_b: Option<Vec<f32>>,
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
            resident: Mutex::new(None),
            class_emb: None,
            pos_emb: None,
            ln_pre_w: None,
            ln_pre_b: None,
        })
    }

    /// Bytes of encoder weights currently in `hipMalloc` (0 until [`Self::ensure_device_resident`]).
    pub fn weight_bytes_on_device(&self) -> u64 {
        self.resident
            .lock()
            .ok()
            .and_then(|g| g.as_ref().map(|w| w.bytes()))
            .unwrap_or(0)
    }

    /// Upload patch_proj (+ bias) and ViT block weights once when `hip_device` matches placement.
    /// No-op (`Ok(0)`) when `hip_device` is `None` or mismatches `self.device`.
    pub fn ensure_device_resident(&self, hip_device: Option<i32>) -> Result<u64, AllpacaError> {
        let Some(want) = self.device else {
            return Ok(0);
        };
        let Some(dev) = hip_device else {
            return Ok(0);
        };
        if want != dev {
            return Ok(0);
        }
        {
            let guard = self.resident.lock().map_err(|_| {
                AllpacaError::Fatal(FatalError::InvalidConfig(
                    "vision resident weights lock poisoned".into(),
                ))
            })?;
            if let Some(w) = guard.as_ref() {
                return Ok(w.bytes());
            }
        }
        let mut slices: Vec<&[f32]> = Vec::with_capacity(2 + self.blocks.len() * 12);
        slices.push(self.weight.as_slice());
        slices.push(self.bias.as_slice());
        for b in self.blocks.iter() {
            slices.push(&b.ln1_w);
            slices.push(&b.ln1_b);
            slices.push(&b.attn_qkv_w);
            slices.push(&b.attn_qkv_b);
            slices.push(&b.attn_proj_w);
            slices.push(&b.attn_proj_b);
            slices.push(&b.ln2_w);
            slices.push(&b.ln2_b);
            slices.push(&b.mlp_fc1_w);
            slices.push(&b.mlp_fc1_b);
            slices.push(&b.mlp_fc2_w);
            slices.push(&b.mlp_fc2_b);
        }
        let uploaded = DeviceWeights::upload(dev, &slices)?;
        let bytes = uploaded.bytes();
        let mut guard = self.resident.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig(
                "vision resident weights lock poisoned".into(),
            ))
        })?;
        *guard = Some(uploaded);
        Ok(bytes)
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
        let conv1 = file.tensor_f32("visual.conv1.weight");
        let has_conv1 = conv1.is_ok();
        let weight = if let Ok(w) = conv1 {
            w
        } else {
            tensor_f32_aliases(
                &file,
                &[
                    "patch_proj.weight",
                    "visual.proj",
                    "visual.proj.weight",
                    "visual.patch_proj.weight",
                ],
            )?
        };
        let bias = if has_conv1 {
            file.tensor_f32("visual.conv1.bias")
                .unwrap_or_else(|_| vec![0.0f32; out_dim])
        } else {
            match tensor_f32_aliases(
                &file,
                &[
                    "patch_proj.bias",
                    "visual.proj.bias",
                    "visual.patch_proj.bias",
                ],
            ) {
                Ok(b) => b,
                Err(_) => vec![0.0f32; out_dim],
            }
        };
        let n_blocks = if n_blocks > 0 {
            n_blocks
        } else {
            count_vit_blocks(&file)
        };
        let mut blocks = Vec::with_capacity(n_blocks as usize);
        for i in 0..n_blocks {
            blocks.push(load_block(&file, out_dim, i as usize)?);
        }
        let mut enc = Self::new(device, out_dim, image_size, patch_size, weight, bias, blocks)?;
        enc.class_emb = file.tensor_f32("visual.class_embedding").ok();
        enc.pos_emb = file.tensor_f32("visual.positional_embedding").ok();
        enc.ln_pre_w = file.tensor_f32("visual.ln_pre.weight").ok();
        enc.ln_pre_b = file.tensor_f32("visual.ln_pre.bias").ok();
        Ok(enc)
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
            let mut tokens = self.patch_embed_tokens(&resized)?;
            if let Some(cls) = self.class_emb.as_ref() {
                if cls.len() == self.out_dim {
                    let mut prefixed = cls.clone();
                    prefixed.extend_from_slice(&tokens);
                    tokens = prefixed;
                }
            }
            if let Some(pos) = self.pos_emb.as_ref() {
                if pos.len() == tokens.len() {
                    for (t, p) in tokens.iter_mut().zip(pos.iter()) {
                        *t += *p;
                    }
                }
            }
            if let (Some(w), Some(b)) = (self.ln_pre_w.as_ref(), self.ln_pre_b.as_ref()) {
                if w.len() == self.out_dim && b.len() == self.out_dim {
                    let seq = tokens.len() / self.out_dim;
                    let mut normed = vec![0.0f32; tokens.len()];
                    for t in 0..seq {
                        let row = &tokens[t * self.out_dim..(t + 1) * self.out_dim];
                        layer_norm(
                            row,
                            w,
                            b,
                            &mut normed[t * self.out_dim..(t + 1) * self.out_dim],
                        );
                    }
                    tokens = normed;
                }
            }
            for (bi, block) in self.blocks.iter().enumerate() {
                let mut used_hip = false;
                if let Ok(guard) = self.resident.lock() {
                    if let Some(weights) = guard.as_ref() {
                        if apply_vit_block_hip(
                            &mut tokens,
                            self.out_dim,
                            self.n_heads,
                            block,
                            weights,
                            self.weight.as_slice(),
                            self.bias.as_slice(),
                            self.blocks.as_slice(),
                            bi,
                        )
                        .is_ok()
                        {
                            used_hip = true;
                        }
                    }
                }
                if !used_hip {
                    apply_vit_block(&mut tokens, self.out_dim, self.n_heads, block)?;
                }
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

    fn ensure_device_resident(&self, hip_device: Option<i32>) -> Result<u64, AllpacaError> {
        PatchVisionEncoder::ensure_device_resident(self, hip_device)
    }

    fn weight_bytes_on_device(&self) -> u64 {
        PatchVisionEncoder::weight_bytes_on_device(self)
    }
}

impl PatchVisionEncoder {
    fn patch_embed_tokens(&self, resized: &RgbImage) -> Result<Vec<f32>, AllpacaError> {
        if let Ok(guard) = self.resident.lock() {
            if let Some(weights) = guard.as_ref() {
                match patch_embed_hip(
                    resized,
                    self.image_size,
                    self.patch_size,
                    self.out_dim,
                    weights,
                    self.bias.as_slice(),
                ) {
                    Ok(tokens) => return Ok(tokens),
                    Err(_) => {
                        // CPU fallback must still match for hosts without a usable HIP path.
                    }
                }
            }
        }
        patch_embed(
            resized,
            self.image_size,
            self.patch_size,
            self.out_dim,
            self.weight.as_slice(),
            self.bias.as_slice(),
        )
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
    let clip = format!("visual.transformer.resblocks.{i}");
    let names = |native: &str, clip_name: &str| -> [String; 2] {
        [format!("{p}.{native}"), format!("{clip}.{clip_name}")]
    };
    let or_zeros = |aliases: &[String], n: usize| -> Result<Vec<f32>, AllpacaError> {
        let refs: Vec<&str> = aliases.iter().map(|s| s.as_str()).collect();
        match tensor_f32_aliases(file, &refs) {
            Ok(v) => Ok(v),
            Err(_) => Ok(vec![0.0f32; n]),
        }
    };
    let req = |aliases: &[String]| -> Result<Vec<f32>, AllpacaError> {
        let refs: Vec<&str> = aliases.iter().map(|s| s.as_str()).collect();
        tensor_f32_aliases(file, &refs)
    };
    let ln1_w = names("ln1.weight", "ln_1.weight");
    let ln1_b = names("ln1.bias", "ln_1.bias");
    let qkv_w = names("attn_qkv.weight", "attn.in_proj_weight");
    let qkv_b = names("attn_qkv.bias", "attn.in_proj_bias");
    let proj_w = names("attn_proj.weight", "attn.out_proj.weight");
    let proj_b = names("attn_proj.bias", "attn.out_proj.bias");
    let ln2_w = names("ln2.weight", "ln_2.weight");
    let ln2_b = names("ln2.bias", "ln_2.bias");
    let fc1_w = names("mlp.fc1.weight", "mlp.c_fc.weight");
    let fc1_b = names("mlp.fc1.bias", "mlp.c_fc.bias");
    let fc2_w = names("mlp.fc2.weight", "mlp.c_proj.weight");
    let fc2_b = names("mlp.fc2.bias", "mlp.c_proj.bias");
    let b = VitBlockWeights {
        ln1_w: or_zeros(&ln1_w, dim)?,
        ln1_b: or_zeros(&ln1_b, dim)?,
        attn_qkv_w: req(&qkv_w)?,
        attn_qkv_b: or_zeros(&qkv_b, 3 * dim)?,
        attn_proj_w: req(&proj_w)?,
        attn_proj_b: or_zeros(&proj_b, dim)?,
        ln2_w: or_zeros(&ln2_w, dim)?,
        ln2_b: or_zeros(&ln2_b, dim)?,
        mlp_fc1_w: req(&fc1_w)?,
        mlp_fc1_b: or_zeros(&fc1_b, mlp_hidden(dim))?,
        mlp_fc2_w: req(&fc2_w)?,
        mlp_fc2_b: or_zeros(&fc2_b, dim)?,
    };
    validate_block(&b, dim, i)?;
    Ok(b)
}

fn count_vit_blocks(file: &SafetensorsFile) -> u32 {
    let mut n = 0u32;
    loop {
        let native = format!("blocks.{n}.attn_qkv.weight");
        let clip = format!("visual.transformer.resblocks.{n}.attn.in_proj_weight");
        if file.tensor(&native).is_some() || file.tensor(&clip).is_some() {
            n += 1;
        } else {
            break;
        }
    }
    n
}

fn tensor_f32_aliases(file: &SafetensorsFile, names: &[&str]) -> Result<Vec<f32>, AllpacaError> {
    let mut last = None;
    for name in names {
        match file.tensor_f32(name) {
            Ok(v) => return Ok(v),
            Err(e) => last = Some(e),
        }
    }
    Err(last.unwrap_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(
            "missing vision tensor (empty alias list)".into(),
        ))
    }))
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
    vit_attention_cpu(&qkv, &mut attn_out, seq, dim, n_heads, head_dim);
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

/// Byte offset of block `block_idx` inside the resident pack (matches [`PatchVisionEncoder::ensure_device_resident`]).
fn resident_block_byte_offset(
    patch_w: &[f32],
    patch_b: &[f32],
    blocks: &[VitBlockWeights],
    block_idx: usize,
) -> usize {
    let mut floats = patch_w.len() + patch_b.len();
    for b in blocks.iter().take(block_idx) {
        floats += vit_block_float_count(b);
    }
    floats.saturating_mul(4)
}

fn vit_block_float_count(b: &VitBlockWeights) -> usize {
    b.ln1_w.len()
        + b.ln1_b.len()
        + b.attn_qkv_w.len()
        + b.attn_qkv_b.len()
        + b.attn_proj_w.len()
        + b.attn_proj_b.len()
        + b.ln2_w.len()
        + b.ln2_b.len()
        + b.mlp_fc1_w.len()
        + b.mlp_fc1_b.len()
        + b.mlp_fc2_w.len()
        + b.mlp_fc2_b.len()
}

/// Device pointers for the four ViT block linear weights (LN tensors stay host).
struct VitBlockLinearPtrs {
    attn_qkv_w: *const f32,
    attn_proj_w: *const f32,
    mlp_fc1_w: *const f32,
    mlp_fc2_w: *const f32,
}

fn vit_block_linear_ptrs(
    weights: &DeviceWeights,
    block_base: usize,
    dim: usize,
) -> VitBlockLinearPtrs {
    let mlp = mlp_hidden(dim);
    let mut off = block_base;
    off += dim.saturating_mul(2).saturating_mul(4); // ln1_w + ln1_b
    let attn_qkv_w = weights.ptr_at(off);
    off += (3 * dim * dim).saturating_mul(4);
    off += (3 * dim).saturating_mul(4); // attn_qkv_b
    let attn_proj_w = weights.ptr_at(off);
    off += (dim * dim).saturating_mul(4);
    off += dim.saturating_mul(4); // attn_proj_b
    off += dim.saturating_mul(2).saturating_mul(4); // ln2_w + ln2_b
    let mlp_fc1_w = weights.ptr_at(off);
    off += (mlp * dim).saturating_mul(4);
    off += mlp.saturating_mul(4); // mlp_fc1_b
    let mlp_fc2_w = weights.ptr_at(off);
    VitBlockLinearPtrs {
        attn_qkv_w,
        attn_proj_w,
        mlp_fc1_w,
        mlp_fc2_w,
    }
}

fn add_bias_rows(out: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    debug_assert_eq!(bias.len(), cols);
    debug_assert_eq!(out.len(), rows * cols);
    for r in 0..rows {
        let row = &mut out[r * cols..(r + 1) * cols];
        for c in 0..cols {
            row[c] += bias[c];
        }
    }
}

/// ViT block with HIP `gemm_nt_dev_b` for linears and `HipWorkspace::attention`
/// for self-attn (softmax on device). LayerNorm stays on CPU: allpaca-hip has
/// RMSNorm only, which is not mean/var LN.
#[allow(clippy::too_many_arguments)]
fn apply_vit_block_hip(
    tokens: &mut [f32],
    dim: usize,
    n_heads: usize,
    block: &VitBlockWeights,
    weights: &DeviceWeights,
    patch_w: &[f32],
    patch_b: &[f32],
    blocks: &[VitBlockWeights],
    block_idx: usize,
) -> Result<(), AllpacaError> {
    let seq = tokens.len() / dim;
    if seq * dim != tokens.len() || seq == 0 {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "vision vit block token shape mismatch".into(),
        )));
    }
    let head_dim = dim / n_heads;
    let mlp = mlp_hidden(dim);
    let base = resident_block_byte_offset(patch_w, patch_b, blocks, block_idx);
    let ptrs = vit_block_linear_ptrs(weights, base, dim);

    let max_a = seq.saturating_mul(dim.max(mlp));
    let max_c = seq.saturating_mul((3 * dim).max(mlp).max(dim));
    let attn_bytes = seq.saturating_mul(dim).saturating_mul(4).saturating_mul(4);
    let scratch = (max_a + max_c)
        .saturating_mul(4)
        .saturating_add(4096)
        .max(attn_bytes.saturating_add(4096));
    let hip = allpaca_hip::HipWorkspace::new(weights.device(), scratch, 1)?;

    let mut normed = vec![0.0f32; tokens.len()];
    for t in 0..seq {
        let row = &tokens[t * dim..(t + 1) * dim];
        layer_norm(row, &block.ln1_w, &block.ln1_b, &mut normed[t * dim..(t + 1) * dim]);
    }

    let mut qkv = vec![0.0f32; seq * 3 * dim];
    hip.gemm_nt_dev_b(
        &normed,
        ptrs.attn_qkv_w,
        &mut qkv,
        seq as i32,
        (3 * dim) as i32,
        dim as i32,
    )?;
    add_bias_rows(&mut qkv, seq, 3 * dim, &block.attn_qkv_b);

    let mut attn_out = vec![0.0f32; seq * dim];
    if vit_attention_hip(&hip, &qkv, &mut attn_out, seq, dim, n_heads, head_dim).is_ok() {
        #[cfg(test)]
        HIP_VIT_ATTN_HITS.fetch_add(1, Ordering::Relaxed);
    } else {
        vit_attention_cpu(&qkv, &mut attn_out, seq, dim, n_heads, head_dim);
    }

    let mut proj = vec![0.0f32; seq * dim];
    hip.gemm_nt_dev_b(
        &attn_out,
        ptrs.attn_proj_w,
        &mut proj,
        seq as i32,
        dim as i32,
        dim as i32,
    )?;
    add_bias_rows(&mut proj, seq, dim, &block.attn_proj_b);
    for i in 0..tokens.len() {
        tokens[i] += proj[i];
    }

    for t in 0..seq {
        let row = &tokens[t * dim..(t + 1) * dim];
        layer_norm(row, &block.ln2_w, &block.ln2_b, &mut normed[t * dim..(t + 1) * dim]);
    }

    let mut hidden = vec![0.0f32; seq * mlp];
    hip.gemm_nt_dev_b(
        &normed,
        ptrs.mlp_fc1_w,
        &mut hidden,
        seq as i32,
        mlp as i32,
        dim as i32,
    )?;
    add_bias_rows(&mut hidden, seq, mlp, &block.mlp_fc1_b);
    for x in hidden.iter_mut() {
        *x = gelu(*x);
    }

    let mut mlp_out = vec![0.0f32; seq * dim];
    hip.gemm_nt_dev_b(
        &hidden,
        ptrs.mlp_fc2_w,
        &mut mlp_out,
        seq as i32,
        dim as i32,
        mlp as i32,
    )?;
    add_bias_rows(&mut mlp_out, seq, dim, &block.mlp_fc2_b);
    for i in 0..tokens.len() {
        tokens[i] += mlp_out[i];
    }
    Ok(())
}

fn vit_attention_cpu(
    qkv: &[f32],
    attn_out: &mut [f32],
    seq: usize,
    dim: usize,
    n_heads: usize,
    head_dim: usize,
) {
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
}

/// Split packed `[seq, 3*dim]` QKV into `[seq, n_heads, head_dim]` layouts and run
/// non-causal HIP attention (device online-softmax).
fn vit_attention_hip(
    hip: &allpaca_hip::HipWorkspace,
    qkv: &[f32],
    attn_out: &mut [f32],
    seq: usize,
    dim: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<(), AllpacaError> {
    let mut q = vec![0.0f32; seq * dim];
    let mut k = vec![0.0f32; seq * dim];
    let mut v = vec![0.0f32; seq * dim];
    for t in 0..seq {
        let base = t * 3 * dim;
        q[t * dim..(t + 1) * dim].copy_from_slice(&qkv[base..base + dim]);
        k[t * dim..(t + 1) * dim].copy_from_slice(&qkv[base + dim..base + 2 * dim]);
        v[t * dim..(t + 1) * dim].copy_from_slice(&qkv[base + 2 * dim..base + 3 * dim]);
    }
    hip.attention(
        &q,
        &k,
        &v,
        attn_out,
        seq as i32,
        seq as i32,
        n_heads as i32,
        n_heads as i32,
        head_dim as i32,
        false,
    )
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

/// Patch linear via resident `DeviceWeights` + HIP `gemm_nt_dev_b` (activations H2D only).
fn patch_embed_hip(
    img: &RgbImage,
    image_size: u32,
    patch_size: u32,
    out_dim: usize,
    weights: &DeviceWeights,
    bias: &[f32],
) -> Result<Vec<f32>, AllpacaError> {
    debug_assert_eq!(img.width, image_size);
    debug_assert_eq!(img.height, image_size);
    let n = (image_size / patch_size) as usize;
    let patch_dim = (patch_size as usize) * (patch_size as usize) * 3;
    let n_tok = n * n;
    let mut a = vec![0.0f32; n_tok * patch_dim];
    for py in 0..n {
        for px in 0..n {
            let token = py * n + px;
            let dst = &mut a[token * patch_dim..(token + 1) * patch_dim];
            let mut i = 0usize;
            for y in 0..patch_size as usize {
                let row = (py * patch_size as usize + y) * image_size as usize * 3;
                for x in 0..patch_size as usize {
                    let off = row + (px * patch_size as usize + x) * 3;
                    dst[i] = img.pixels[off] as f32 / 255.0;
                    dst[i + 1] = img.pixels[off + 1] as f32 / 255.0;
                    dst[i + 2] = img.pixels[off + 2] as f32 / 255.0;
                    i += 3;
                }
            }
        }
    }
    let mut c = vec![0.0f32; n_tok * out_dim];
    let scratch = (a.len() + c.len()).saturating_mul(4).saturating_add(4096);
    let hip = allpaca_hip::HipWorkspace::new(weights.device(), scratch, 1)?;
    hip.gemm_nt_dev_b(
        &a,
        weights.ptr_at(0),
        &mut c,
        n_tok as i32,
        out_dim as i32,
        patch_dim as i32,
    )?;
    for t in 0..n_tok {
        let row = &mut c[t * out_dim..(t + 1) * out_dim];
        for o in 0..out_dim {
            row[o] += bias[o];
        }
    }
    Ok(c)
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
    fn ensure_device_resident_noops_without_hip() {
        let enc = PatchVisionEncoder::synthetic(Some(0), 32, 1).unwrap();
        assert_eq!(enc.weight_bytes_on_device(), 0);
        assert_eq!(enc.ensure_device_resident(None).unwrap(), 0);
        assert_eq!(enc.weight_bytes_on_device(), 0);
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_resident_encode_matches_cpu_patch_embed() {
        if allpaca_hip::device_count() == 0 {
            eprintln!("skip hip_resident_encode_matches_cpu_patch_embed: no HIP device");
            return;
        }
        let jpeg = tiny_jpeg();
        let cpu = PatchVisionEncoder::synthetic_sized(None, 32, 64, 16, 11, 0).unwrap();
        let hip = PatchVisionEncoder::synthetic_sized(Some(0), 32, 64, 16, 11, 0).unwrap();
        let bytes = hip.ensure_device_resident(Some(0)).unwrap();
        assert!(bytes > 0, "expected weight upload");
        let a = cpu
            .encode(&[ImageInput::new(jpeg.clone(), "image/jpeg")])
            .unwrap();
        let b = hip
            .encode(&[ImageInput::new(jpeg, "image/jpeg")])
            .unwrap();
        assert_eq!(a.n_tokens, b.n_tokens);
        assert_eq!(a.embeddings.len(), b.embeddings.len());
        for (i, (x, y)) in a.embeddings.iter().zip(b.embeddings.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-3,
                "mismatch at {i}: cpu={x} hip={y}"
            );
        }
    }

    #[cfg(feature = "hip")]
    #[test]
    fn vit_block_hip_encode_matches_cpu() {
        if allpaca_hip::device_count() == 0 {
            eprintln!("skip vit_block_hip_encode_matches_cpu: no HIP device");
            return;
        }
        let jpeg = tiny_jpeg();
        let cpu = PatchVisionEncoder::synthetic_sized(None, 32, 64, 16, 13, 1).unwrap();
        let hip = PatchVisionEncoder::synthetic_sized(Some(0), 32, 64, 16, 13, 1).unwrap();
        assert_eq!(hip.n_blocks(), 1);
        let bytes = hip.ensure_device_resident(Some(0)).unwrap();
        assert!(bytes > 0, "expected weight upload including vit block");
        HIP_VIT_ATTN_HITS.store(0, Ordering::Relaxed);
        let a = cpu
            .encode(&[ImageInput::new(jpeg.clone(), "image/jpeg")])
            .unwrap();
        let b = hip
            .encode(&[ImageInput::new(jpeg, "image/jpeg")])
            .unwrap();
        assert_eq!(a.n_tokens, b.n_tokens);
        assert_eq!(a.embeddings.len(), b.embeddings.len());
        for (i, (x, y)) in a.embeddings.iter().zip(b.embeddings.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-3,
                "mismatch at {i}: cpu={x} hip={y}"
            );
        }
        assert!(
            HIP_VIT_ATTN_HITS.load(Ordering::Relaxed) >= 1,
            "expected HIP attention (device softmax) when resident"
        );
    }

    #[cfg(feature = "hip")]
    #[test]
    fn vit_attention_hip_matches_cpu_on_packed_qkv() {
        if allpaca_hip::device_count() == 0 {
            eprintln!("skip vit_attention_hip_matches_cpu_on_packed_qkv: no HIP device");
            return;
        }
        let seq = 4usize;
        let n_heads = 2usize;
        let head_dim = 4usize;
        let dim = n_heads * head_dim;
        let mut qkv = vec![0.0f32; seq * 3 * dim];
        for (i, x) in qkv.iter_mut().enumerate() {
            *x = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        let mut cpu_out = vec![0.0f32; seq * dim];
        let mut hip_out = vec![0.0f32; seq * dim];
        vit_attention_cpu(&qkv, &mut cpu_out, seq, dim, n_heads, head_dim);
        let scratch = (seq * dim * 4 * 4).saturating_add(4096);
        let hip = allpaca_hip::HipWorkspace::new(0, scratch, 1).unwrap();
        vit_attention_hip(&hip, &qkv, &mut hip_out, seq, dim, n_heads, head_dim)
            .expect("HIP vit attention");
        for (i, (c, h)) in cpu_out.iter().zip(hip_out.iter()).enumerate() {
            assert!(
                (c - h).abs() < 1e-3,
                "attn mismatch at {i}: cpu={c} hip={h}"
            );
        }
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
    fn load_clip_style_aliases_matches_native() {
        let out_dim = 32usize;
        let patch_dim = (DEFAULT_PATCH_SIZE as usize) * (DEFAULT_PATCH_SIZE as usize) * 3;
        let mlp = mlp_hidden(out_dim);
        let mut state = 21u64 ^ 0xC0FFEE_u64;
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
        let block = synthetic_block(out_dim, &mut next).unwrap();
        let enc = PatchVisionEncoder::new(
            None,
            out_dim,
            DEFAULT_IMAGE_SIZE,
            DEFAULT_PATCH_SIZE,
            weight.clone(),
            bias.clone(),
            vec![block.clone()],
        )
        .unwrap();
        let tensors = vec![
            SafetensorsWriteTensor {
                name: "visual.proj".into(),
                dims: vec![out_dim as u64, patch_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &weight,
            },
            SafetensorsWriteTensor {
                name: "visual.proj.bias".into(),
                dims: vec![out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &bias,
            },
            SafetensorsWriteTensor {
                name: "visual.transformer.resblocks.0.ln_1.weight".into(),
                dims: vec![out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &block.ln1_w,
            },
            SafetensorsWriteTensor {
                name: "visual.transformer.resblocks.0.ln_1.bias".into(),
                dims: vec![out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &block.ln1_b,
            },
            SafetensorsWriteTensor {
                name: "visual.transformer.resblocks.0.attn.in_proj_weight".into(),
                dims: vec![(3 * out_dim) as u64, out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &block.attn_qkv_w,
            },
            SafetensorsWriteTensor {
                name: "visual.transformer.resblocks.0.attn.in_proj_bias".into(),
                dims: vec![(3 * out_dim) as u64],
                dtype: SafetensorsDtype::F32,
                data: &block.attn_qkv_b,
            },
            SafetensorsWriteTensor {
                name: "visual.transformer.resblocks.0.attn.out_proj.weight".into(),
                dims: vec![out_dim as u64, out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &block.attn_proj_w,
            },
            SafetensorsWriteTensor {
                name: "visual.transformer.resblocks.0.attn.out_proj.bias".into(),
                dims: vec![out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &block.attn_proj_b,
            },
            SafetensorsWriteTensor {
                name: "visual.transformer.resblocks.0.ln_2.weight".into(),
                dims: vec![out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &block.ln2_w,
            },
            SafetensorsWriteTensor {
                name: "visual.transformer.resblocks.0.ln_2.bias".into(),
                dims: vec![out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &block.ln2_b,
            },
            SafetensorsWriteTensor {
                name: "visual.transformer.resblocks.0.mlp.c_fc.weight".into(),
                dims: vec![mlp as u64, out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &block.mlp_fc1_w,
            },
            SafetensorsWriteTensor {
                name: "visual.transformer.resblocks.0.mlp.c_fc.bias".into(),
                dims: vec![mlp as u64],
                dtype: SafetensorsDtype::F32,
                data: &block.mlp_fc1_b,
            },
            SafetensorsWriteTensor {
                name: "visual.transformer.resblocks.0.mlp.c_proj.weight".into(),
                dims: vec![out_dim as u64, mlp as u64],
                dtype: SafetensorsDtype::F32,
                data: &block.mlp_fc2_w,
            },
            SafetensorsWriteTensor {
                name: "visual.transformer.resblocks.0.mlp.c_proj.bias".into(),
                dims: vec![out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &block.mlp_fc2_b,
            },
        ];
        let path = std::env::temp_dir().join("allpaca_vision_clip_alias_test.safetensors");
        write_safetensors(&path, &tensors).unwrap();
        let loaded =
            PatchVisionEncoder::load(&path, None, out_dim, DEFAULT_IMAGE_SIZE, DEFAULT_PATCH_SIZE, 1)
                .unwrap();
        assert_eq!(loaded.n_blocks(), 1);
        let jpeg = tiny_jpeg();
        let a = enc
            .encode(&[ImageInput::new(jpeg.clone(), "image/jpeg")])
            .unwrap();
        let btok = loaded
            .encode(&[ImageInput::new(jpeg, "image/jpeg")])
            .unwrap();
        assert_eq!(a.embeddings, btok.embeddings);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_clip_checkpoint_conv1_class_pos_auto_blocks() {
        let out_dim = 32usize;
        let patch = DEFAULT_PATCH_SIZE;
        let image_size = DEFAULT_IMAGE_SIZE;
        let n_patches = ((image_size / patch) as usize).pow(2);
        let enc = PatchVisionEncoder::synthetic_sized(None, out_dim, image_size, patch, 7, 1)
            .unwrap();
        let block = enc.blocks[0].clone();
        let conv1 = enc.weight.as_slice().to_vec();
        let cls = vec![0.25f32; out_dim];
        let mut pos = vec![0.0f32; (n_patches + 1) * out_dim];
        for (i, p) in pos.iter_mut().enumerate() {
            *p = (i as f32) * 0.001;
        }
        let ln_w = vec![1.0f32; out_dim];
        let ln_b = vec![0.0f32; out_dim];
        let tensors = vec![
            SafetensorsWriteTensor {
                name: "visual.conv1.weight".into(),
                dims: vec![out_dim as u64, 3, patch as u64, patch as u64],
                dtype: SafetensorsDtype::F32,
                data: &conv1,
            },
            SafetensorsWriteTensor {
                name: "visual.class_embedding".into(),
                dims: vec![out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &cls,
            },
            SafetensorsWriteTensor {
                name: "visual.positional_embedding".into(),
                dims: vec![(n_patches + 1) as u64, out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &pos,
            },
            SafetensorsWriteTensor {
                name: "visual.ln_pre.weight".into(),
                dims: vec![out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &ln_w,
            },
            SafetensorsWriteTensor {
                name: "visual.ln_pre.bias".into(),
                dims: vec![out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &ln_b,
            },
            SafetensorsWriteTensor {
                name: "visual.transformer.resblocks.0.attn.in_proj_weight".into(),
                dims: vec![(3 * out_dim) as u64, out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &block.attn_qkv_w,
            },
            SafetensorsWriteTensor {
                name: "visual.transformer.resblocks.0.attn.out_proj.weight".into(),
                dims: vec![out_dim as u64, out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &block.attn_proj_w,
            },
            SafetensorsWriteTensor {
                name: "visual.transformer.resblocks.0.mlp.c_fc.weight".into(),
                dims: vec![mlp_hidden(out_dim) as u64, out_dim as u64],
                dtype: SafetensorsDtype::F32,
                data: &block.mlp_fc1_w,
            },
            SafetensorsWriteTensor {
                name: "visual.transformer.resblocks.0.mlp.c_proj.weight".into(),
                dims: vec![out_dim as u64, mlp_hidden(out_dim) as u64],
                dtype: SafetensorsDtype::F32,
                data: &block.mlp_fc2_w,
            },
        ];
        let path = std::env::temp_dir().join("allpaca_vision_clip_ckpt.safetensors");
        write_safetensors(&path, &tensors).unwrap();
        let loaded =
            PatchVisionEncoder::load(&path, None, out_dim, image_size, patch, 0).unwrap();
        assert_eq!(loaded.n_blocks(), 1, "n_blocks=0 must auto-count resblocks");
        let jpeg = tiny_jpeg();
        let tokens = loaded
            .encode(&[ImageInput::new(jpeg, "image/jpeg")])
            .unwrap();
        assert_eq!(
            tokens.n_tokens,
            n_patches + 1,
            "CLIP class token prepends one soft token"
        );
        assert_eq!(tokens.embeddings.len(), tokens.n_tokens * out_dim);
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
