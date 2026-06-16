//! Image-generation backend contract.
//!
//! A [`Backend`] turns a prompt + seed into an [`Image`]. Concrete backends
//! (e.g. Stable Diffusion via `candle`) live behind this trait so the
//! selection loop in [`crate::best_of_n`] stays independent of any model.

use crate::Result;

/// A generated raster image: RGB8, row-major, `width * height * 3` bytes.
///
/// Kept intentionally minimal — encoding to PNG/JPEG is the CLI's job, not the
/// library's. Downstream consumers take the raw buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// RGB8 pixel data, length `width * height * 3`.
    pub rgb: Vec<u8>,
}

impl Image {
    /// Construct an image, validating that the buffer length matches the
    /// declared dimensions.
    pub fn new(width: u32, height: u32, rgb: Vec<u8>) -> Result<Self> {
        let expected = width as usize * height as usize * 3;
        if rgb.len() != expected {
            return Err(crate::Error::Backend(format!(
                "buffer length {} does not match {width}x{height} RGB ({expected} bytes)",
                rgb.len()
            )));
        }
        Ok(Self { width, height, rgb })
    }

    /// Save the image as a PNG. Available with the `clip`/`sd` features (which
    /// pull in the `image` crate); the base build has no image encoder.
    #[cfg(any(feature = "clip", feature = "sd"))]
    pub fn save_png(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let buf = image::RgbImage::from_raw(self.width, self.height, self.rgb.clone())
            .ok_or_else(|| crate::Error::Backend("image buffer size mismatch".into()))?;
        buf.save_with_format(path.as_ref(), image::ImageFormat::Png)
            .map_err(|e| crate::Error::Backend(format!("saving PNG: {e}")))
    }
}

/// A source of generated images.
///
/// Given a prompt and a seed, produce one image. The same `(prompt, seed)`
/// should yield the same image for a deterministic backend — that determinism
/// is what makes the selection loop reproducible and testable.
pub trait Backend {
    /// Generate a single image for `prompt` using `seed`.
    fn generate(&self, prompt: &str, seed: u64) -> Result<Image>;
}
