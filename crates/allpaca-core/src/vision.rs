//! CPU patch-embed vision encoder with loadable F32 weights.
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

/// Linear patch-embed → soft tokens for LM prefill.
pub struct PatchVisionEncoder {
    device: Option<i32>,
    out_dim: usize,
    image_size: u32,
    patch_size: u32,
    /// Row-major `[out_dim, patch_dim]` where `patch_dim = patch*patch*3`.
    weight: Arc<Vec<f32>>,
    bias: Arc<Vec<f32>>,
}

impl PatchVisionEncoder {
    pub fn new(
        device: Option<i32>,
        out_dim: usize,
        image_size: u32,
        patch_size: u32,
        weight: Vec<f32>,
        bias: Vec<f32>,
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
        Ok(Self {
            device,
            out_dim,
            image_size,
            patch_size,
            weight: Arc::new(weight),
            bias: Arc::new(bias),
        })
    }

    /// Deterministic synthetic weights for tests / CPU-only synthetic Engine.
    pub fn synthetic(device: Option<i32>, out_dim: usize, seed: u64) -> Result<Self, AllpacaError> {
        Self::synthetic_sized(device, out_dim, DEFAULT_IMAGE_SIZE, DEFAULT_PATCH_SIZE, seed)
    }

    pub fn synthetic_sized(
        device: Option<i32>,
        out_dim: usize,
        image_size: u32,
        patch_size: u32,
        seed: u64,
    ) -> Result<Self, AllpacaError> {
        let patch_dim = (patch_size as usize) * (patch_size as usize) * 3;
        let mut weight = vec![0.0f32; out_dim * patch_dim];
        let mut bias = vec![0.0f32; out_dim];
        let mut state = seed ^ 0xC0FFEE_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            ((state >> 33) as f32) * (1.0 / (u32::MAX as f32)) - 0.5
        };
        for w in weight.iter_mut() {
            *w = next() * 0.02;
        }
        for b in bias.iter_mut() {
            *b = next() * 0.01;
        }
        Self::new(device, out_dim, image_size, patch_size, weight, bias)
    }

    pub fn load(
        path: &Path,
        device: Option<i32>,
        out_dim: usize,
        image_size: u32,
        patch_size: u32,
    ) -> Result<Self, AllpacaError> {
        let file = SafetensorsFile::open(path)?;
        let weight = file.tensor_f32("patch_proj.weight")?;
        let bias = match file.tensor_f32("patch_proj.bias") {
            Ok(b) => b,
            Err(_) => vec![0.0f32; out_dim],
        };
        Self::new(device, out_dim, image_size, patch_size, weight, bias)
    }

    pub fn write_safetensors(&self, path: &Path) -> Result<(), AllpacaError> {
        let patch_dim = (self.patch_size as usize) * (self.patch_size as usize) * 3;
        write_safetensors(
            path,
            &[
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
            ],
        )
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    pub fn n_patches(&self) -> usize {
        let n = (self.image_size / self.patch_size) as usize;
        n * n
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
            let tokens = patch_embed(
                &resized,
                self.image_size,
                self.patch_size,
                self.out_dim,
                self.weight.as_slice(),
                self.bias.as_slice(),
            )?;
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

    #[test]
    fn synthetic_encoder_roundtrips_jpeg_bytes() {
        let enc = PatchVisionEncoder::synthetic(Some(1), 32, 7).unwrap();
        assert_eq!(enc.n_patches(), 16);
        // 1x1 red JPEG
        let jpeg = {
            let img = image::RgbImage::from_pixel(8, 8, image::Rgb([200, 20, 20]));
            let mut buf = Vec::new();
            image::DynamicImage::ImageRgb8(img)
                .write_to(
                    &mut std::io::Cursor::new(&mut buf),
                    image::ImageFormat::Jpeg,
                )
                .unwrap();
            buf
        };
        let tokens = enc
            .encode(&[ImageInput::new(jpeg, "image/jpeg")])
            .unwrap();
        assert_eq!(tokens.n_tokens, 16);
        assert_eq!(tokens.embeddings.len(), 16 * 32);
        assert_eq!(tokens.device, Some(1));
        assert!(tokens.embeddings.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn load_written_weights_matches_synthetic() {
        let enc = PatchVisionEncoder::synthetic(None, 32, 99).unwrap();
        let path = std::env::temp_dir().join("allpaca_vision_patch_test.safetensors");
        enc.write_safetensors(&path).unwrap();
        let loaded = PatchVisionEncoder::load(&path, Some(0), 32, DEFAULT_IMAGE_SIZE, DEFAULT_PATCH_SIZE)
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
