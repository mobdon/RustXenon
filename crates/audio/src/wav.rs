//! .wav files: [`WavWriter`] writes 16-bit PCM, [`read`] reads it back
//! (16-bit PCM or 32-bit float) for comparisons.

use std::io::{self, Seek, SeekFrom, Write};

/// A [`Sink`](crate::Sink) writing 16-bit PCM. Sizes in the header are
/// patched by [`WavWriter::finish`]; a dropped, unfinished writer leaves a
/// file whose header says 0 bytes of data.
pub struct WavWriter<W: Write + Seek> {
    out: W,
    channels: u16,
    data_bytes: u32,
}

impl WavWriter<io::BufWriter<std::fs::File>> {
    pub fn create(path: impl AsRef<std::path::Path>, rate: u32, channels: u16) -> io::Result<Self> {
        Self::new(
            io::BufWriter::new(std::fs::File::create(path)?),
            rate,
            channels,
        )
    }
}

impl<W: Write + Seek> WavWriter<W> {
    pub fn new(mut out: W, rate: u32, channels: u16) -> io::Result<Self> {
        let align = channels * 2;
        out.write_all(b"RIFF\0\0\0\0WAVEfmt ")?;
        out.write_all(&16u32.to_le_bytes())?;
        out.write_all(&1u16.to_le_bytes())?; // PCM
        out.write_all(&channels.to_le_bytes())?;
        out.write_all(&rate.to_le_bytes())?;
        out.write_all(&(rate * align as u32).to_le_bytes())?;
        out.write_all(&align.to_le_bytes())?;
        out.write_all(&16u16.to_le_bytes())?;
        out.write_all(b"data\0\0\0\0")?;
        Ok(Self {
            out,
            channels,
            data_bytes: 0,
        })
    }

    /// Interleaved samples in -1..=1 (clamped).
    pub fn write_samples(&mut self, s: &[f32]) -> io::Result<()> {
        let mut b = Vec::with_capacity(s.len() * 2);
        for &x in s {
            b.extend(((x.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
        }
        self.out.write_all(&b)?;
        self.data_bytes = self
            .data_bytes
            .checked_add(b.len() as u32)
            .ok_or_else(|| io::Error::other("wav file over 4 GiB"))?;
        Ok(())
    }

    /// Patches the header sizes and returns the writer.
    pub fn finish(mut self) -> io::Result<W> {
        self.out.seek(SeekFrom::Start(4))?;
        self.out.write_all(&(36 + self.data_bytes).to_le_bytes())?;
        self.out.seek(SeekFrom::Start(40))?;
        self.out.write_all(&self.data_bytes.to_le_bytes())?;
        self.out.seek(SeekFrom::End(0))?;
        self.out.flush()?;
        Ok(self.out)
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }
}

impl<W: Write + Seek> crate::Sink for WavWriter<W> {
    fn write(&mut self, stereo: &[f32]) -> io::Result<()> {
        self.write_samples(stereo)
    }
}

/// A decoded .wav: rate, channel count, interleaved samples.
#[derive(Debug)]
pub struct Wav {
    pub rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

/// Reads a 16-bit PCM or 32-bit float .wav.
pub fn read(bytes: &[u8]) -> Result<Wav, String> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let u16_at = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
    let (mut fmt, mut p) = (None, 12);
    while p + 8 <= bytes.len() {
        let len = u32::from_le_bytes(bytes[p + 4..p + 8].try_into().unwrap()) as usize;
        let body = bytes
            .get(p + 8..p + 8 + len)
            .ok_or("chunk runs past the end")?;
        match &bytes[p..p + 4] {
            b"fmt " if len >= 16 => {
                let rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                fmt = Some((u16_at(p + 8), u16_at(p + 10), rate, u16_at(p + 22)));
            }
            b"data" => {
                let (tag, channels, rate, bits) = fmt.ok_or("data before fmt")?;
                let samples = match (tag, bits) {
                    (1, 16) => body
                        .chunks_exact(2)
                        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32767.0)
                        .collect(),
                    (3, 32) => body
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect(),
                    _ => return Err(format!("unsupported wav format {tag}/{bits} bits")),
                };
                return Ok(Wav {
                    rate,
                    channels,
                    samples,
                });
            }
            _ => {}
        }
        p += 8 + len + (len & 1);
    }
    Err("no data chunk".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let s: Vec<f32> = (0..1000).map(|i| ((i as f32) * 0.01).sin() * 0.8).collect();
        let mut w = WavWriter::new(io::Cursor::new(Vec::new()), 48_000, 2).unwrap();
        w.write_samples(&s[..300]).unwrap();
        w.write_samples(&s[300..]).unwrap();
        let bytes = w.finish().unwrap().into_inner();
        assert_eq!(bytes.len(), 44 + 2000);
        assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), 2000);
        let back = read(&bytes).unwrap();
        assert_eq!(
            (back.rate, back.channels, back.samples.len()),
            (48_000, 2, 1000)
        );
        assert!(s
            .iter()
            .zip(&back.samples)
            .all(|(a, b)| (a - b).abs() < 1e-4));
    }

    #[test]
    fn rejects_junk() {
        assert!(read(b"nope").is_err());
        assert!(read(b"RIFF\0\0\0\0WAVEdata\x10\0\0\0").is_err());
    }
}
