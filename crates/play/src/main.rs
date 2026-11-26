//! `rustxenon-play`: the double-click launcher. Lists the built games under
//! its home folder, starts one as a separate process and waits for it.
//!
//! ```text
//! rustxenon-play [--home <dir>] [<built game folder>...]
//! rustxenon-play --screenshot <out.png> [--page games|settings] [--home <dir>]
//! ```
//!
//! Home is the launcher's own folder unless `--home` says otherwise. A folder
//! on the command line (a folder dropped on the exe in Explorer) is added to
//! the recent list. It writes its log to `rustxenon-play-launcher.log`
//! beside itself.

#![windows_subsystem = "windows"]

#[cfg(not(target_os = "android"))]
mod desktop;

#[cfg(not(target_os = "android"))]
fn main() {
    desktop::main()
}

#[cfg(target_os = "android")]
fn main() {}
