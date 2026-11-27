//! Presented frames and PNG output.

use std::path::Path;

use crate::Error;

/// One presented frame: `rgba8` is `width * height * 4` bytes, rows top to
/// bottom, R G B A per pixel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

impl Frame {
    /// The frame as an 8-bit RGBA PNG file.
    pub fn to_png_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut out = Vec::new();
        let mut enc = png::Encoder::new(&mut out, self.width, self.height);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let png_err = |e: png::EncodingError| Error::Png(e.to_string());
        let mut w = enc.write_header().map_err(png_err)?;
        w.write_image_data(&self.rgba8).map_err(png_err)?;
        w.finish().map_err(png_err)?;
        Ok(out)
    }

    /// Writes [`Frame::to_png_bytes`] to `path`.
    pub fn write_png(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        std::fs::write(path, self.to_png_bytes()?).map_err(|e| Error::Io(e.to_string()))
    }
}

/// The pixels, so a shared frame can be shown without copying them.
impl AsRef<[u8]> for Frame {
    fn as_ref(&self) -> &[u8] {
        &self.rgba8
    }
}
