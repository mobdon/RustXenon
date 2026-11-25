//! Frame bundles: a few frames of commands and the guest memory they read,
//! saved by the host and replayed offline (the drawer's `replay_frame`
//! example). Skipping a range of draws in the last frame finds the draw
//! behind a wrong colour. A bundle holds game data: keep it local.

use crate::{Command, SliceMemory};
use std::io;
use std::path::Path;

/// The command list, as JSON.
pub const COMMANDS: &str = "commands.json";
/// Guest physical memory from address 0, raw.
pub const MEMORY: &str = "memory.bin";

pub fn write_commands(dir: &Path, cmds: &[Command]) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut f = io::BufWriter::new(std::fs::File::create(dir.join(COMMANDS))?);
    serde_json::to_writer(&mut f, cmds).map_err(io::Error::other)?;
    io::Write::flush(&mut f)
}

pub fn read(dir: &Path) -> io::Result<(Vec<Command>, SliceMemory)> {
    let f = io::BufReader::new(std::fs::File::open(dir.join(COMMANDS))?);
    let cmds = serde_json::from_reader(f).map_err(io::Error::other)?;
    let data = std::fs::read(dir.join(MEMORY))?;
    Ok((cmds, SliceMemory { base: 0, data }))
}

/// Index of the first command of the last frame: just after the
/// second-to-last swap (0 with fewer than two swaps).
pub fn last_frame_start(cmds: &[Command]) -> usize {
    let mut swaps = cmds
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, c)| matches!(c, Command::Swap { .. }));
    swaps.next();
    swaps.next().map_or(0, |(i, _)| i + 1)
}

/// The commands up to the last swap, with draws `lo..=hi` of the last frame
/// left out, counted from 1 as the capture's draw list counts them. Returns
/// the commands and the number of draws in the last frame.
pub fn skip_draws(cmds: &[Command], lo: u32, hi: u32) -> (Vec<Command>, u32) {
    // Commands after the last swap draw nothing that frame shows.
    let cmds = match cmds.iter().rposition(|c| matches!(c, Command::Swap { .. })) {
        Some(i) => &cmds[..=i],
        None => cmds,
    };
    let start = last_frame_start(cmds);
    let mut n = 0;
    let mut out = cmds[..start].to_vec();
    for c in &cmds[start..] {
        if matches!(c, Command::Draw(_)) {
            n += 1;
            if (lo..=hi).contains(&n) {
                continue;
            }
        }
        out.push(c.clone());
    }
    (out, n)
}

/// The commands up to and including draw `k` of the last frame (counted
/// from 1), to look at the render targets as that draw left them.
pub fn cut_after_draw(cmds: &[Command], k: u32) -> Vec<Command> {
    let start = last_frame_start(&skip_draws(cmds, 1, 0).0);
    let mut n = 0;
    let end = cmds[start..]
        .iter()
        .position(|c| {
            n += u32::from(matches!(c, Command::Draw(_)));
            n == k && matches!(c, Command::Draw(_))
        })
        .map_or(cmds.len(), |i| start + i + 1);
    cmds[..end].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Draw, IndexSource, PrimitiveType};

    fn draw(n: u32) -> Command {
        Command::Draw(Draw {
            primitive: PrimitiveType::TriangleList,
            index_count: n,
            source: IndexSource::Auto,
            initiator: 0,
            viz_query: None,
        })
    }

    fn swap() -> Command {
        Command::Swap {
            signature_ok: true,
            frontbuffer: 0,
            width: 1,
            height: 1,
        }
    }

    #[test]
    fn skips_draws_of_the_last_frame_only() {
        let mut cmds = vec![draw(1), swap(), draw(2), draw(3), draw(4), draw(5), swap()];
        let whole = cmds.clone();
        // A draw of the next frame, kept with the span that held the swap.
        cmds.push(draw(6));
        let (out, n) = skip_draws(&cmds, 2, 3);
        assert_eq!(n, 4);
        assert_eq!(out, vec![draw(1), swap(), draw(2), draw(5), swap()]);
        assert_eq!(skip_draws(&cmds, 1, 0).0, whole);
    }

    #[test]
    fn cuts_after_a_draw_of_the_last_frame() {
        let cmds = vec![draw(1), swap(), draw(2), swap(), draw(3), draw(4), swap()];
        assert_eq!(cut_after_draw(&cmds, 1), cmds[..5].to_vec());
        assert_eq!(cut_after_draw(&cmds, 2), cmds[..6].to_vec());
        assert_eq!(cut_after_draw(&cmds, 9), cmds);
    }

    #[test]
    fn bundle_round_trips() {
        let dir = std::env::temp_dir().join(format!("xenon-bundle-{}", std::process::id()));
        let cmds = vec![
            Command::WriteRegisters {
                base: 0,
                values: vec![1, 2, 3],
            },
            draw(3),
            swap(),
        ];
        write_commands(&dir, &cmds).unwrap();
        std::fs::write(dir.join(MEMORY), [7u8; 16]).unwrap();
        let (back, mem) = read(&dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(back, cmds);
        assert_eq!(mem.data, vec![7u8; 16]);
    }
}
