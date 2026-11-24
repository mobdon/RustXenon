// Register bitfields from Xenia src/xenia/gpu/registers.h (PA_SU_SC_MODE_CNTL,
// PA_SU_VTX_CNTL, PA_CL_VTE_CNTL, PA_SC_WINDOW_*, PA_SC_SCREEN_SCISSOR_*,
// RB_COLOR_MASK, RB_BLENDCONTROL, RB_DEPTHCONTROL) and enums from
// src/xenia/gpu/xenos.h (BlendFactor, BlendOp, CompareFunction). The scissor
// follows GetScissor and the viewport offsets GetHostViewportInfo in
// src/xenia/gpu/draw_util.cc; the position epilogue follows the end of the
// vertex shader in src/xenia/gpu/spirv_shader_translator.cc; depth writes
// need the depth test as in src/xenia/gpu/d3d12/pipeline_cache.cc
// (BSD 3-Clause, see CREDITS.md).

//! Fixed-function state decoded from the register file.

use xenon_gpu::regs;
use xenon_gpu::RegisterFile;

fn sign_extend(v: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((v << shift) as i32) >> shift
}

/// `xenos::CompareFunction` (3 bits). Note its order differs from the
/// WAIT_REG_MEM compare codes.
pub fn compare(f: u32) -> wgpu::CompareFunction {
    use wgpu::CompareFunction as C;
    match f & 7 {
        0 => C::Never,
        1 => C::Less,
        2 => C::Equal,
        3 => C::LessEqual,
        4 => C::Greater,
        5 => C::NotEqual,
        6 => C::GreaterEqual,
        _ => C::Always,
    }
}

fn blend_factor(f: u32) -> Option<wgpu::BlendFactor> {
    use wgpu::BlendFactor as B;
    Some(match f {
        0 => B::Zero,
        1 => B::One,
        4 => B::Src,
        5 => B::OneMinusSrc,
        6 => B::SrcAlpha,
        7 => B::OneMinusSrcAlpha,
        8 => B::Dst,
        9 => B::OneMinusDst,
        10 => B::DstAlpha,
        11 => B::OneMinusDstAlpha,
        12 => B::Constant,
        13 => B::OneMinusConstant,
        // wgpu has one constant colour; the alpha variants read its alpha
        // only through the alpha blend channel, so this is exact there and
        // approximate for colour.
        14 => B::Constant,
        15 => B::OneMinusConstant,
        16 => B::SrcAlphaSaturated,
        _ => return None,
    })
}

fn blend_component(src: u32, op: u32, dst: u32) -> Option<wgpu::BlendComponent> {
    use wgpu::BlendOperation as O;
    let operation = match op {
        0 => O::Add,
        1 => O::Subtract,
        2 => O::Min,
        3 => O::Max,
        4 => O::ReverseSubtract,
        _ => return None,
    };
    let (src_factor, dst_factor) = match operation {
        // WebGPU requires One factors with min and max, which ignore them.
        O::Min | O::Max => (wgpu::BlendFactor::One, wgpu::BlendFactor::One),
        _ => (blend_factor(src)?, blend_factor(dst)?),
    };
    Some(wgpu::BlendComponent {
        src_factor,
        dst_factor,
        operation,
    })
}

/// RB_BLENDCONTROL: `Ok(None)` when it is the identity (src One, dst Zero,
/// add), `Err` for a value outside the Xenos enums.
pub fn blend(reg: u32) -> Result<Option<wgpu::BlendState>, String> {
    let f = |lo: u32, n: u32| (reg >> lo) & ((1 << n) - 1);
    let color = blend_component(f(0, 5), f(5, 3), f(8, 5));
    let alpha = blend_component(f(16, 5), f(21, 3), f(24, 5));
    match (color, alpha) {
        (Some(color), Some(alpha)) => {
            let state = wgpu::BlendState { color, alpha };
            Ok((state != wgpu::BlendState::REPLACE).then_some(state))
        }
        _ => Err(format!("blend control 0x{reg:08X}")),
    }
}

/// RB_COLOR_MASK bits for render target `rt`.
pub fn write_mask(color_mask: u32, rt: u32) -> wgpu::ColorWrites {
    wgpu::ColorWrites::from_bits_truncate((color_mask >> (4 * rt)) & 0xF)
}

/// RB_DEPTHCONTROL, depth part.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Depth {
    pub test: bool,
    pub write: bool,
    pub compare: wgpu::CompareFunction,
    pub stencil: bool,
}

pub fn depth(reg: u32) -> Depth {
    let test = reg & 2 != 0;
    Depth {
        test,
        // z_write_enable only counts while z_enable is set: with the depth
        // test off nothing is written (Xenia d3d12/pipeline_cache.cc, the
        // depth/stencil state of GetCurrentStateDescription).
        write: test && reg & 4 != 0,
        compare: if test {
            compare(reg >> 4)
        } else {
            wgpu::CompareFunction::Always
        },
        stencil: reg & 1 != 0,
    }
}

impl Depth {
    /// Whether a depth attachment is needed at all.
    pub fn used(self) -> bool {
        self.test || self.write || self.stencil
    }
}

/// PA_SU_SC_MODE_CNTL: front face and culled face. `None` when both faces
/// are culled (the draw produces nothing, as Xenia skips it).
pub fn cull(mode_cntl: u32) -> Option<(wgpu::FrontFace, Option<wgpu::Face>)> {
    // face bit 2: 0 front is CCW, 1 front is CW (in window coordinates,
    // which this drawer keeps y-down like the host framebuffer).
    let front = if mode_cntl & 4 != 0 {
        wgpu::FrontFace::Cw
    } else {
        wgpu::FrontFace::Ccw
    };
    match mode_cntl & 3 {
        0 => Some((front, None)),
        1 => Some((front, Some(wgpu::Face::Front))),
        2 => Some((front, Some(wgpu::Face::Back))),
        _ => None,
    }
}

/// Window offset (PA_SC_WINDOW_OFFSET, two 15-bit signed fields).
fn window_offset(r: &RegisterFile) -> (i32, i32) {
    let v = r.get(regs::PA_SC_WINDOW_OFFSET);
    (sign_extend(v, 15), sign_extend(v >> 16, 15))
}

/// Pixel rectangle `(x, y, width, height)`.
pub type PixelRect = (u32, u32, u32, u32);

/// The draw scissor as Xenia's GetScissor computes it, clamped to a render
/// target of `width` x `height`.
pub fn scissor(r: &RegisterFile, width: u32, height: u32) -> PixelRect {
    let tl = r.get(regs::PA_SC_WINDOW_SCISSOR_TL);
    let br = r.get(regs::PA_SC_WINDOW_SCISSOR_BR);
    let (mut x0, mut y0) = ((tl & 0x3FFF) as i32, ((tl >> 16) & 0x3FFF) as i32);
    let (mut x1, mut y1) = ((br & 0x3FFF) as i32, ((br >> 16) & 0x3FFF) as i32);
    if tl & (1 << 31) == 0 {
        let (ox, oy) = window_offset(r);
        x0 += ox;
        y0 += oy;
        x1 += ox;
        y1 += oy;
    }
    let stl = r.get(regs::PA_SC_SCREEN_SCISSOR_TL);
    let sbr = r.get(regs::PA_SC_SCREEN_SCISSOR_BR);
    x0 = x0.max(sign_extend(stl, 15));
    y0 = y0.max(sign_extend(stl >> 16, 15));
    x1 = x1.min(sign_extend(sbr, 15));
    y1 = y1.min(sign_extend(sbr >> 16, 15));
    let x0 = x0.clamp(0, width as i32);
    let y0 = y0.clamp(0, height as i32);
    let x1 = x1.clamp(x0, width as i32);
    let y1 = y1.clamp(y0, height as i32);
    (x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32)
}

/// Uniform for the vertex epilogue the drawer appends to every vertex
/// shader (`XeDraw` in WGSL): guest position -> host clip space for a host
/// viewport covering the whole render target.
pub fn epilogue(r: &RegisterFile, width: u32, height: u32) -> [u32; 12] {
    let vte = r.get(regs::PA_CL_VTE_CNTL);
    let get = |bit: u32, reg: u32, default: f32| {
        if vte & bit != 0 {
            f32::from_bits(r.get(reg))
        } else {
            default
        }
    };
    let (sx, ox) = (
        get(1, regs::PA_CL_VPORT_XSCALE, 1.0),
        get(2, regs::PA_CL_VPORT_XOFFSET, 0.0),
    );
    let (sy, oy) = (
        get(4, regs::PA_CL_VPORT_YSCALE, 1.0),
        get(8, regs::PA_CL_VPORT_YOFFSET, 0.0),
    );
    let (sz, oz) = (
        get(16, regs::PA_CL_VPORT_ZSCALE, 1.0),
        get(32, regs::PA_CL_VPORT_ZOFFSET, 0.0),
    );
    // Window offset and the Direct3D 9 pixel centre (PA_SU_VTX_CNTL
    // pix_center 0: centres at .0, so host = guest + 0.5), as Xenia's
    // GetHostViewportInfo with its default half_pixel_offset.
    let (mut ax, mut ay) = (0.0, 0.0);
    if r.get(regs::PA_SU_SC_MODE_CNTL) & (1 << 16) != 0 {
        let (wx, wy) = window_offset(r);
        ax += wx as f32;
        ay += wy as f32;
    }
    if r.get(regs::PA_SU_VTX_CNTL) & 1 == 0 {
        ax += 0.5;
        ay += 0.5;
    }
    let (w, h) = (width as f32, height as f32);
    // Window x -> NDC: x * 2 / W - 1; window y (down) -> NDC y (up):
    // 1 - y * 2 / H. Depth is already 0..1 in the Direct3D clip space.
    let scale = [sx * 2.0 / w, -sy * 2.0 / h, sz, 0.0];
    let offset = [
        (ox + ax) * 2.0 / w - 1.0,
        1.0 - (oy + ay) * 2.0 / h,
        oz,
        0.0,
    ];
    let mut out = [0u32; 12];
    for i in 0..4 {
        out[i] = scale[i].to_bits();
        out[4 + i] = offset[i].to_bits();
    }
    // vtx_xy_fmt, vtx_z_fmt, vtx_w0_fmt (bits 8, 9, 10).
    out[8] = (vte >> 8) & 1;
    out[9] = (vte >> 9) & 1;
    out[10] = (vte >> 10) & 1;
    out
}

/// WGSL appended to every vertex shader; `out.pos = o_pos;` becomes
/// `out.pos = xe_epilogue(o_pos);`.
pub const EPILOGUE_WGSL: &str = "
struct XeDraw {
    ndc_scale: vec4<f32>,
    ndc_offset: vec4<f32>,
    flags: vec4<u32>,
}

@group(0) @binding(2) var<uniform> xe_draw: XeDraw;

// Guest position -> host clip space. flags: x = XY already divided by W,
// y = Z already divided by W, z = W is W (else 1/W), as PA_CL_VTE_CNTL.
fn xe_epilogue(p: vec4<f32>) -> vec4<f32> {
    var w = p.w;
    if (xe_draw.flags.z == 0u) {
        w = 1.0 / w;
    }
    var xy = p.xy;
    if (xe_draw.flags.x != 0u) {
        xy = xy * w;
    }
    var z = p.z;
    if (xe_draw.flags.y != 0u) {
        z = z * w;
    }
    let xyz = vec3<f32>(xy, z) * xe_draw.ndc_scale.xyz + xe_draw.ndc_offset.xyz * w;
    return vec4<f32>(xyz, w);
}
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_identity_is_none_and_alpha_blend_maps() {
        // src One, add, dst Zero for colour and alpha.
        assert_eq!(blend(0x0001_0001), Ok(None));
        let b = blend(6 | (7 << 8) | (1 << 16) | (7 << 24))
            .unwrap()
            .unwrap();
        assert_eq!(b.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(b.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(b.alpha.src_factor, wgpu::BlendFactor::One);
        assert!(blend(2).is_err());
    }

    #[test]
    fn depth_and_cull_decode() {
        let d = depth(2 | 4 | (3 << 4));
        assert!(d.test && d.write && !d.stencil);
        assert_eq!(d.compare, wgpu::CompareFunction::LessEqual);
        assert!(!depth(0).used());
        // Depth writes without the depth test write nothing.
        assert!(!depth(4).write && !depth(4).used());
        assert_eq!(cull(0), Some((wgpu::FrontFace::Ccw, None)));
        assert_eq!(
            cull(2 | 4),
            Some((wgpu::FrontFace::Cw, Some(wgpu::Face::Back)))
        );
        assert_eq!(cull(3), None);
    }

    #[test]
    fn scissor_applies_window_offset_and_screen_scissor() {
        let mut r = RegisterFile::default();
        r.set(regs::PA_SC_WINDOW_SCISSOR_TL, 2 | (3 << 16));
        r.set(regs::PA_SC_WINDOW_SCISSOR_BR, 50 | (60 << 16));
        r.set(regs::PA_SC_WINDOW_OFFSET, 10 | (0x7FFF << 16)); // x +10, y -1
        r.set(regs::PA_SC_SCREEN_SCISSOR_BR, 40 | (8192 << 16));
        assert_eq!(scissor(&r, 64, 64), (12, 2, 28, 57));
        r.set(regs::PA_SC_WINDOW_SCISSOR_TL, (1 << 31) | 2 | (3 << 16));
        assert_eq!(scissor(&r, 64, 32), (2, 3, 38, 29));
    }

    #[test]
    fn epilogue_maps_window_corners_to_ndc() {
        // VTE off, OpenGL pixel centre: window (0, 0) -> NDC (-1, 1).
        let mut r = RegisterFile::default();
        r.set(regs::PA_SU_VTX_CNTL, 1);
        r.set(regs::PA_CL_VTE_CNTL, 1 << 10);
        let e = epilogue(&r, 64, 32);
        let f = |i: usize| f32::from_bits(e[i]);
        assert_eq!((f(4), f(5)), (-1.0, 1.0));
        assert_eq!((64.0 * f(0) + f(4), 32.0 * f(1) + f(5)), (1.0, -1.0));
        assert_eq!(e[10], 1);
    }
}
