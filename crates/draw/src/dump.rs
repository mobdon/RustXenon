//! Opt-in shader dump for diagnostics (`XENON_SHADER_DUMP`).
//!
//! Once per program hash and stage it writes to the dump directory
//! `<hash>.<vs|ps>.ucode` (the raw words, little-endian), `<hash>.<vs|ps>.txt`
//! (the [`xenon_shader::disasm`] listing) and `<hash>.<vs|ps>.wgsl` (the
//! translation, or the error), and for each pixel shader
//! `<hash>.ps.draw.txt` with the state of the first draw that used it.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::PathBuf;

use xenon_gpu::regs::{self, RegisterFile};
use xenon_shader::disasm::{disassemble, Disassembly};
use xenon_shader::ucode::FetchInstr;
use xenon_shader::{ShaderStage, PIXEL_FLOAT_CONST_BASE};

use crate::{state, texture_log_line, TargetFormat};

/// Where and which shaders to dump.
#[derive(Clone, Debug)]
pub struct ShaderDump {
    pub dir: PathBuf,
    /// Only these pixel shaders (with the vertex shaders drawn with them)
    /// and vertex shaders; every shader when `None`.
    pub only: Option<HashSet<u64>>,
    /// (hash, is pixel) already written.
    done: HashSet<(u64, bool)>,
}

impl ShaderDump {
    pub fn new(dir: impl Into<PathBuf>, only: Option<HashSet<u64>>) -> Self {
        ShaderDump {
            dir: dir.into(),
            only,
            done: HashSet::new(),
        }
    }

    /// Parses `<dir>[,<hash>...]` (hashes in hex, as the logs print them).
    pub fn parse(v: &str) -> Result<Self, String> {
        let mut parts = v.split(',').map(str::trim);
        let dir = parts
            .next()
            .filter(|d| !d.is_empty())
            .ok_or("no directory")?;
        let hashes = parts
            .map(|h| {
                let hex = h.trim_start_matches("0x").trim_start_matches("0X");
                u64::from_str_radix(hex, 16).map_err(|_| format!("{h:?} is not a hex hash"))
            })
            .collect::<Result<HashSet<u64>, String>>()?;
        Ok(Self::new(dir, (!hashes.is_empty()).then_some(hashes)))
    }

    /// Dumps what is new of a draw's shaders. `vs_wgsl` / `ps_wgsl` are the
    /// translations or the translation errors.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draw(
        &mut self,
        regs: &RegisterFile,
        vs: (u64, &[u32]),
        vs_wgsl: Result<&str, String>,
        ps: (u64, &[u32]),
        ps_wgsl: Result<&str, String>,
        draw: &xenon_gpu::Draw,
    ) -> std::io::Result<()> {
        let want_ps = self.only.as_ref().is_none_or(|o| o.contains(&ps.0));
        let want_vs = want_ps || self.only.as_ref().is_some_and(|o| o.contains(&vs.0));
        let new_vs = want_vs && !self.done.contains(&(vs.0, false));
        let new_ps = want_ps && !self.done.contains(&(ps.0, true));
        if !new_vs && !new_ps {
            return Ok(());
        }
        std::fs::create_dir_all(&self.dir)?;
        let vs_d = disassemble(vs.1, ShaderStage::Vertex);
        if new_vs {
            self.program("vs", vs, &vs_d, vs_wgsl)?;
            self.done.insert((vs.0, false));
        }
        if new_ps {
            let ps_d = disassemble(ps.1, ShaderStage::Pixel);
            self.program("ps", ps, &ps_d, ps_wgsl)?;
            let text = draw_state(regs, vs.0, &vs_d, ps.0, &ps_d, draw);
            std::fs::write(self.dir.join(format!("{:016x}.ps.draw.txt", ps.0)), text)?;
            self.done.insert((ps.0, true));
        }
        Ok(())
    }

    fn program(
        &self,
        stage: &str,
        (hash, words): (u64, &[u32]),
        d: &Disassembly,
        wgsl: Result<&str, String>,
    ) -> std::io::Result<()> {
        let path = |ext: &str| self.dir.join(format!("{hash:016x}.{stage}.{ext}"));
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(path("ucode"), bytes)?;
        std::fs::write(path("txt"), &d.text)?;
        let wgsl = wgsl.map_or_else(|e| format!("// translation failed: {e}\n"), str::to_owned);
        std::fs::write(path("wgsl"), wgsl)
    }
}

/// The state of a draw that the shaders read or that changes their output.
fn draw_state(
    r: &RegisterFile,
    vs: u64,
    vs_d: &Disassembly,
    ps: u64,
    ps_d: &Disassembly,
    draw: &xenon_gpu::Draw,
) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "vs {vs:016x} ps {ps:016x}");
    let _ = writeln!(
        s,
        "draw {:?} {} indices {:?}",
        draw.primitive, draw.index_count, draw.source
    );

    let _ = writeln!(s, "\n# vertex fetches");
    let mut seen = 0u128;
    for f in &vs_d.fetches {
        if let FetchInstr::Vertex(v) = f {
            let k = v.fetch_const;
            let base = regs::SHADER_CONSTANT_FETCH + 2 * k;
            let _ = writeln!(s, "{}", xenon_shader::disasm::fetch_text(f));
            if seen & (1 << k) != 0 {
                continue;
            }
            seen |= 1 << k;
            let _ = writeln!(
                s,
                "    vf{k} {:?} | {:08x} {:08x}",
                r.vertex_fetch(k),
                r.get(base),
                r.get(base + 1)
            );
        }
    }

    let _ = writeln!(s, "\n# texture fetches");
    for (stage, d) in [("vs", vs_d), ("ps", ps_d)] {
        let mut seen = 0u32;
        for f in &d.fetches {
            let FetchInstr::Texture(t) = f else { continue };
            let _ = writeln!(s, "{stage} {}", xenon_shader::disasm::fetch_text(f));
            if seen & (1 << t.slot) == 0 {
                seen |= 1 << t.slot;
                let explicit =
                    !t.use_computed_lod || t.use_register_lod || t.use_register_gradients;
                let c = r.fetch_constant(t.slot);
                let line = texture_log_line(ps, t.slot, c, explicit, t.lod_bias != 0);
                let _ = writeln!(s, "    {line}");
            }
        }
    }

    let _ = writeln!(s, "\n# render state");
    for (name, reg) in [
        ("RB_SURFACE_INFO", regs::RB_SURFACE_INFO),
        ("RB_MODECONTROL", regs::RB_MODECONTROL),
        ("RB_COLOR_MASK", regs::RB_COLOR_MASK),
        ("RB_DEPTHCONTROL", regs::RB_DEPTHCONTROL),
        ("RB_DEPTH_INFO", regs::RB_DEPTH_INFO),
    ] {
        let _ = writeln!(s, "{name} {:08x}", r.get(reg));
    }
    // The targets the pixel shader exports to.
    for rt in (0..4).filter(|rt| ps_d.text.contains(&format!("oC{rt}."))) {
        let info = r.get(regs::RB_COLOR_INFO_RT[rt]);
        let blend = r.get(regs::RB_BLENDCONTROL_RT[rt]);
        let f = (info >> 16) & 0xF;
        let _ = writeln!(
            s,
            "RB_COLOR_INFO[{rt}] {info:08x} format {f} {:?} exp_bias {} base {} | RB_BLENDCONTROL {blend:08x} {:?}",
            TargetFormat::color(f),
            (info << 6) as i32 >> 26,
            info & 0xFFF,
            state::blend(blend),
        );
    }

    let _ = writeln!(s, "\n# float constants read");
    for (stage, base, d) in [("vs", 0, vs_d), ("ps", PIXEL_FLOAT_CONST_BASE, ps_d)] {
        let all: Vec<u32>;
        let list = if d.float_consts_relative {
            all = (0..256).collect();
            &all
        } else {
            &d.float_consts
        };
        for &i in list {
            let w: [u32; 4] = std::array::from_fn(|j| {
                r.get(regs::SHADER_CONSTANT_FLOAT + (base + i) * 4 + j as u32)
            });
            let _ = writeln!(
                s,
                "{stage} c{i} {:?} | {:08x} {:08x} {:08x} {:08x}",
                w.map(f32::from_bits),
                w[0],
                w[1],
                w[2],
                w[3]
            );
        }
    }
    let words = |start: u32, n: u32| -> String {
        (start..start + n)
            .map(|a| format!("{:08x}", r.get(a)))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let _ = writeln!(
        s,
        "\n# bool constants\n{}",
        words(regs::SHADER_CONSTANT_BOOL, 8)
    );
    let _ = writeln!(
        s,
        "# loop constants\n{}",
        words(regs::SHADER_CONSTANT_LOOP, 32)
    );
    s
}
