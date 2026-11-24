// The resolve rectangle follows GetResolveInfo in Xenia
// src/xenia/gpu/draw_util.cc (three vertices in vertex fetch constant 0, the
// half-pixel adjustment, the top-left rounding, the scissor clamp, the
// 8-pixel alignment and the surface pitch clamp), and the destination
// addressing follows its 2D tiled copy destination (each source pixel
// (x, y) lands at Tiled2D(x, y) from RB_COPY_DEST_BASE)
// (BSD 3-Clause, see CREDITS.md).

//! EDRAM resolve helpers that do not touch the GPU.

use xenon_gpu::regs;
use xenon_gpu::{FetchConstantType, GuestRead, RegisterFile, Resolve};
use xenon_texture::{endian, tile};

use crate::state;

/// Resolve rectangle, right/bottom exclusive, in render target pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl Rect {
    pub fn width(&self) -> u32 {
        self.x1 - self.x0
    }
    pub fn height(&self) -> u32 {
        self.y1 - self.y0
    }
    pub fn is_empty(&self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }
}

/// D3D11's float -> 16.8 fixed point rule (round to nearest even, NaN to
/// 0, saturating), as the rasterizer snaps vertices.
fn fixed_16p8(v: f32) -> i32 {
    if v.is_nan() {
        return 0;
    }
    (v * 256.0)
        .round_ties_even()
        .clamp(-(1 << 23) as f32, ((1 << 23) - 1) as f32) as i32
}

/// The rectangle a resolve copies and clears. Like Xenia it comes from the
/// three rectangle vertices in vertex fetch constant 0 (written by the
/// console's Direct3D runtime). When that fetch constant is not a 6-dword
/// vertex buffer, or its memory is unreadable, the window scissor is used
/// instead and the returned flag is false.
pub fn rect(
    r: &Resolve,
    regs_now: &RegisterFile,
    mem: &dyn GuestRead,
    target_height: u32,
) -> (Rect, bool) {
    let pitch = r.surface_pitch;
    let (sx, sy, sw, sh) = state::scissor(regs_now, u32::MAX >> 2, u32::MAX >> 2);
    let (sx1, sy1) = (sx + sw, sy + sh);
    let vf = r.vertex_fetch0;
    let verts = (vf.fetch_type == FetchConstantType::Vertex && vf.size_dwords == 6)
        .then(|| mem.read_bytes(vf.address, 24))
        .flatten();
    let (mut x0, mut y0, mut x1, mut y1, from_vertices) = match verts {
        Some(bytes) => {
            let half = if regs_now.get(regs::PA_SU_VTX_CNTL) & 1 == 0 {
                0.5
            } else {
                0.0
            };
            let v: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|c| {
                    let bits = vf.endian.from_guest_bytes([c[0], c[1], c[2], c[3]]);
                    fixed_16p8(f32::from_bits(bits) + half)
                })
                .collect();
            let round = |a: i32| (a + 127) >> 8;
            let (mut x0, mut y0) = (
                round(v[0].min(v[2]).min(v[4])),
                round(v[1].min(v[3]).min(v[5])),
            );
            let (mut x1, mut y1) = (
                round(v[0].max(v[2]).max(v[4])),
                round(v[1].max(v[3]).max(v[5])),
            );
            if r.vtx_window_offset_enable {
                x0 += r.window_offset_x;
                x1 += r.window_offset_x;
                y0 += r.window_offset_y;
                y1 += r.window_offset_y;
            }
            (x0, y0, x1, y1, true)
        }
        None => (0, 0, i32::MAX, i32::MAX, false),
    };
    let clamp = |v: i32, lo: u32, hi: u32| v.clamp(lo as i32, hi as i32);
    x0 = clamp(x0, sx, sx1);
    x1 = clamp(x1, sx, sx1);
    y0 = clamp(y0, sy, sy1);
    y1 = clamp(y1, sy, sy1);
    // Direct3D 9 aligns resolves to 8 pixels; Xenia expands to that.
    x0 &= !7;
    y0 &= !7;
    x1 = (x1 + 7) & !7;
    y1 = (y1 + 7) & !7;
    let pitch_aligned = (pitch & !7) as i32;
    let max_y = target_height as i32;
    let rect = Rect {
        x0: x0.min(pitch_aligned) as u32,
        y0: y0.min(max_y) as u32,
        x1: x1.min(pitch_aligned).max(0) as u32,
        y1: y1.min(max_y).max(0) as u32,
    };
    (rect, from_vertices)
}

/// Writes `rgba8` (the rectangle's pixels, rows top to bottom, R in the
/// low byte of each little-endian texel) into the 32x32-tiled 8_8_8_8
/// surface at `dest_base` with pitch `dest_pitch` (rounded up to 32), byte
/// swapped by `dest_endian` (0..=3, as `xenos::Endian`).
pub fn write_tiled(
    mem: &mut dyn crate::GuestMemory,
    dest_base: u32,
    dest_pitch: u32,
    rect: Rect,
    rgba8: &[u8],
    dest_endian: u32,
) -> Result<(), String> {
    let pitch = tile::align_to_tile(dest_pitch);
    let extent = tile::tiled_extent_2d(rect.x1, rect.y1, pitch, 4)
        .map_err(|e| format!("resolve destination: {e}"))?;
    let mut buf = mem
        .read_bytes(dest_base, extent as usize)
        .ok_or_else(|| format!("resolve destination 0x{dest_base:08X} is not readable"))?;
    let e = xenon_texture::Endian::from_bits(dest_endian);
    let width = rect.width() as usize;
    for (i, px) in rgba8.chunks_exact(4).enumerate() {
        let x = rect.x0 + (i % width) as u32;
        let y = rect.y0 + (i / width) as u32;
        let at = tile::tiled_offset_2d(x, y, pitch, 2) as usize;
        let mut t = [px[0], px[1], px[2], px[3]];
        endian::swap_in_place(&mut t, e);
        buf[at..at + 4].copy_from_slice(&t);
    }
    if mem.write_bytes(dest_base, &buf) {
        Ok(())
    } else {
        Err(format!(
            "resolve destination 0x{dest_base:08X} is not writable"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xenon_gpu::SliceMemory;

    fn regs_with_scissor() -> RegisterFile {
        let mut r = RegisterFile::default();
        r.set(regs::PA_SC_WINDOW_SCISSOR_TL, 1 << 31);
        r.set(regs::PA_SC_WINDOW_SCISSOR_BR, 100 | (50 << 16));
        r.set(regs::PA_SC_SCREEN_SCISSOR_BR, 8192 | (8192 << 16));
        r
    }

    #[test]
    fn rect_from_vertex_fetch_0() {
        let mut r = regs_with_scissor();
        r.set(regs::RB_SURFACE_INFO, 80);
        // Three D3D9 corners (-0.5 offsets), big-endian floats, k8in32.
        let verts = [-0.5f32, -0.5, 63.5, -0.5, 63.5, 19.5];
        let words: Vec<u32> = verts.iter().map(|f| f.to_bits()).collect();
        let mem = SliceMemory::from_words(0x1000, &words);
        r.set(regs::SHADER_CONSTANT_FETCH, 0x1000 | 3);
        r.set(regs::SHADER_CONSTANT_FETCH + 1, (6 << 2) | 2);
        let res = Resolve::from_registers(&r);
        let (rect, from_vertices) = rect(&res, &r, &mem, 2048);
        assert!(from_vertices);
        assert_eq!(
            rect,
            Rect {
                x0: 0,
                y0: 0,
                x1: 64,
                y1: 24
            }
        );
    }

    #[test]
    fn rect_falls_back_to_scissor_and_pitch() {
        let mut r = regs_with_scissor();
        r.set(regs::RB_SURFACE_INFO, 64);
        let res = Resolve::from_registers(&r);
        let mem = SliceMemory::default();
        let (rect, from_vertices) = rect(&res, &r, &mem, 2048);
        assert!(!from_vertices);
        assert_eq!(
            rect,
            Rect {
                x0: 0,
                y0: 0,
                x1: 64,
                y1: 56
            }
        );
    }

    #[test]
    fn write_tiled_round_trips_through_detile() {
        let rect = Rect {
            x0: 0,
            y0: 0,
            x1: 8,
            y1: 8,
        };
        let px: Vec<u8> = (0..64u32).flat_map(|i| [i as u8, 1, 2, 3]).collect();
        let mut mem = SliceMemory {
            base: 0,
            data: vec![0; 32 * 32 * 4],
        };
        write_tiled(&mut mem, 0, 8, rect, &px, 2).unwrap();
        let mut raw = mem.data.clone();
        endian::swap_in_place(&mut raw, xenon_texture::Endian::K8in32);
        assert_eq!(tile::detile_2d(&raw, 8, 8, 32, 4).unwrap(), px);
    }
}
