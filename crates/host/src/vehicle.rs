//! Whether the player is in a vehicle, read from the game's own script
//! natives, so the keyboard's vehicle keys (W gas, S brake) follow the game.
//!
//! A title's script machine calls each native through a table of function
//! pointers with one argument, a call record: the result pointer at +0 and
//! the argument array at +8. Two natives, named by role in the game's
//! `hooks.toml` (baked in by `build.rs`), are wrapped at run time, so the
//! generated code needs no rebuild:
//!
//! - `script_player_char` (player index, out character handle): remembers
//!   the player's character handle.
//! - `script_in_vehicle` (character handle, returns a flag): when the
//!   character is the player's, hands the result to the window.
//!
//! Scripts ask the second one about the player many times a second; the
//! window drops the answer when it goes stale. A game without the roles
//! keeps the keys on foot.

use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::OnceLock;
use xenon_rt::{Context, GuestFn, Runtime};

include!(concat!(env!("OUT_DIR"), "/games.rs"));

/// The player's character handle, as the last `script_player_char` gave it.
static PLAYER: AtomicU32 = AtomicU32::new(u32::MAX);
/// The last answer for the player, for the log: 0 none yet, 1 no, 2 yes.
static LAST: AtomicU8 = AtomicU8::new(0);
static PLAYER_CHAR: OnceLock<GuestFn> = OnceLock::new();
static IN_VEHICLE: OnceLock<GuestFn> = OnceLock::new();

/// The two roles' addresses in the hook file `text`, when it has both.
fn roles(text: &str, sha256: &str) -> Option<(u32, u32)> {
    let h = xenon_recomp::hooks::Hooks::parse(text, sha256).ok()?;
    Some((
        *h.hooks.get("script_player_char")?,
        *h.hooks.get("script_in_vehicle")?,
    ))
}

/// Wraps the game's natives, when its hook file names them. Call before
/// [`Runtime::share`], after the game's functions are registered.
pub fn install(rt: &mut Runtime, sha256: &str) {
    let Some((player, vehicle)) = GAMES
        .iter()
        .find(|g| g.0.eq_ignore_ascii_case(sha256))
        .and_then(|g| roles(g.1, sha256))
    else {
        return;
    };
    let (Some(p), Some(v)) = (rt.function(player), rt.function(vehicle)) else {
        crate::log("vehicle keys: the hook file names functions this build lacks");
        return;
    };
    let _ = PLAYER_CHAR.set(p);
    let _ = IN_VEHICLE.set(v);
    rt.set_override(player, player_char);
    rt.set_override(vehicle, in_vehicle);
}

/// The record's argument `i`.
fn arg(rt: &Runtime, record: u32, i: u32) -> u32 {
    rt.mem.read_u32(rt.mem.read_u32(record + 8) + 4 * i)
}

fn player_char(rt: &Runtime, c: &mut Context) {
    let record = c.r[3] as u32;
    (PLAYER_CHAR.get().unwrap())(rt, c);
    PLAYER.store(rt.mem.read_u32(arg(rt, record, 1)), Ordering::Relaxed);
}

fn in_vehicle(rt: &Runtime, c: &mut Context) {
    let record = c.r[3] as u32;
    let who = arg(rt, record, 0);
    (IN_VEHICLE.get().unwrap())(rt, c);
    if who != PLAYER.load(Ordering::Relaxed) {
        return;
    }
    let on = rt.mem.read_u32(rt.mem.read_u32(record)) != 0;
    if LAST.swap(1 + on as u8, Ordering::Relaxed) != 1 + on as u8 {
        crate::log(&format!(
            "player {} a vehicle",
            if on { "in" } else { "not in" }
        ));
    }
    if let Some(w) = crate::kernel::input::WINDOW.get() {
        w.set_vehicle(on);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xenon_rt::GuestMemory;

    const SHA: &str = "00000000000000000000000000000000000000000000000000000000000000aa";

    #[test]
    fn reads_both_roles_or_none() {
        let both = format!(
            "sha256 = \"{SHA}\"\n[hooks]\nscript_player_char = 0x82000100\n\
             script_in_vehicle = 0x82000200\nframe_present = 0x82000300\n"
        );
        assert_eq!(roles(&both, SHA), Some((0x8200_0100, 0x8200_0200)));
        let one = "[hooks]\nscript_in_vehicle = 0x82000200\n";
        assert_eq!(roles(one, SHA), None);
        assert_eq!(roles(&both, &SHA.replace("aa", "bb")), None); // another game's file
        assert_eq!(roles("not toml [", SHA), None);
    }

    /// Stand-ins for the game's natives: the player is character 7 and
    /// sits in a vehicle; everyone else is on foot.
    fn game_player_char(rt: &Runtime, c: &mut Context) {
        rt.mem.write_u32(arg(rt, c.r[3] as u32, 1), 7);
        c.r[3] = 0xdead;
    }
    fn game_in_vehicle(rt: &Runtime, c: &mut Context) {
        let r = c.r[3] as u32;
        let on = arg(rt, r, 0) == 7;
        rt.mem.write_u32(rt.mem.read_u32(r), on as u32);
        c.r[3] = 0xdead;
    }

    #[test]
    fn wraps_the_natives_and_watches_only_the_player() {
        let mut rt = Runtime::new(GuestMemory::new(1 << 16));
        rt.register(0x8200_0100, game_player_char);
        rt.register(0x8200_0200, game_in_vehicle);
        let _ = PLAYER_CHAR.set(game_player_char);
        let _ = IN_VEHICLE.set(game_in_vehicle);
        rt.set_override(0x8200_0100, player_char);
        rt.set_override(0x8200_0200, in_vehicle);
        // A call record at 0x100: result at 0x200, arguments at 0x300.
        let m = &rt.mem;
        m.write_u32(0x100, 0x200);
        m.write_u32(0x108, 0x300);
        m.write_u32(0x304, 0x400); // where the player's handle goes
        let mut c = Context::new();
        c.r[3] = 0x100;
        rt.call(&mut c, 0x8200_0100);
        assert_eq!(PLAYER.load(Ordering::Relaxed), 7);
        // Someone else on foot: the game's answer passes through, unwatched.
        m.write_u32(0x300, 9);
        c.r[3] = 0x100;
        rt.call(&mut c, 0x8200_0200);
        assert_eq!((m.read_u32(0x200), c.r[3]), (0, 0xdead));
        assert_eq!(LAST.load(Ordering::Relaxed), 0);
        // The player, in a vehicle.
        m.write_u32(0x300, 7);
        c.r[3] = 0x100;
        rt.call(&mut c, 0x8200_0200);
        assert_eq!(m.read_u32(0x200), 1);
        assert_eq!(LAST.load(Ordering::Relaxed), 2);
    }
}
