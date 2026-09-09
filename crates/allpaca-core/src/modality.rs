//! Multimodal encoder placement + encode API.
//!
//! Config + scheduler expose `vision_device`; real vision weights are still deferred.
//! `NoopVisionEncoder` is a placement-aware stub used until a checkpoint encoder lands.

use allpaca_types::{AllpacaError, DegradedError};

/// Where a modality encoder should run (vision/audio).
pub trait EncoderPlacement: Send + Sync {
    fn device(&self) -> Option<i32>;
}

/// Raw image bytes for a multimodal Request (URL fetch is a Server concern).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInput {
    pub bytes: Vec<u8>,
    pub mime: String,
}

impl ImageInput {
    pub fn new(bytes: impl Into<Vec<u8>>, mime: impl Into<String>) -> Self {
        Self {
            bytes: bytes.into(),
            mime: mime.into(),
        }
    }
}

/// Soft tokens produced by a vision encoder for LM prefill injection.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionTokens {
    pub embeddings: Vec<f32>,
    pub n_tokens: usize,
    pub device: Option<i32>,
}

impl VisionTokens {
    pub fn empty(device: Option<i32>) -> Self {
        Self {
            embeddings: Vec::new(),
            n_tokens: 0,
            device,
        }
    }
}

/// Vision encoder: place on `vision_device`, produce soft tokens for the Target Model.
pub trait VisionEncoder: EncoderPlacement {
    fn encode(&self, images: &[ImageInput]) -> Result<VisionTokens, AllpacaError>;
}

/// Stub vision encoder: placement only until real weights land.
#[derive(Debug, Clone, Default)]
pub struct NoopVisionEncoder {
    device: Option<i32>,
}

impl NoopVisionEncoder {
    pub fn new(device: Option<i32>) -> Self {
        Self { device }
    }
}

impl EncoderPlacement for NoopVisionEncoder {
    fn device(&self) -> Option<i32> {
        self.device
    }
}

impl VisionEncoder for NoopVisionEncoder {
    fn encode(&self, images: &[ImageInput]) -> Result<VisionTokens, AllpacaError> {
        if images.is_empty() {
            return Ok(VisionTokens::empty(self.device));
        }
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "vision encoder runtime not loaded (noop)".into(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_reports_configured_device() {
        let enc = NoopVisionEncoder::new(Some(1));
        assert_eq!(enc.device(), Some(1));
        let none = NoopVisionEncoder::new(None);
        assert_eq!(none.device(), None);
    }

    #[test]
    fn noop_encode_empty_is_ok() {
        let enc = NoopVisionEncoder::new(Some(1));
        let tokens = enc.encode(&[]).unwrap();
        assert_eq!(tokens.n_tokens, 0);
        assert_eq!(tokens.device, Some(1));
    }

    #[test]
    fn encode_reports_configured_vision_device() {
        struct PlaceEnc(Option<i32>);
        impl EncoderPlacement for PlaceEnc {
            fn device(&self) -> Option<i32> {
                self.0
            }
        }
        impl VisionEncoder for PlaceEnc {
            fn encode(&self, images: &[ImageInput]) -> Result<VisionTokens, AllpacaError> {
                Ok(VisionTokens {
                    embeddings: vec![0.0; images.len()],
                    n_tokens: images.len(),
                    device: self.0,
                })
            }
        }
        let enc = PlaceEnc(Some(1));
        let tokens = enc
            .encode(&[ImageInput::new(vec![0], "image/png")])
            .unwrap();
        assert_eq!(tokens.device, Some(1));
        assert_eq!(tokens.device, enc.device());
        assert_eq!(tokens.n_tokens, 1);
    }

    #[test]
    fn noop_encode_images_are_degraded() {
        let enc = NoopVisionEncoder::new(Some(1));
        let err = enc
            .encode(&[ImageInput::new(vec![1], "image/png")])
            .unwrap_err();
        assert_eq!(err.class(), allpaca_types::ErrorClass::Degraded);
    }
}
