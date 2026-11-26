//! Gives the executable its icon: an icon resource file handed to the MSVC
//! linker. Other targets get no file icon. `xenon-recomp` copies this file
//! into each generated game, so the game gets the same icon.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let var = |k: &str| std::env::var(k).unwrap_or_default();
    if var("CARGO_CFG_TARGET_OS") != "windows" || var("CARGO_CFG_TARGET_ENV") != "msvc" {
        return;
    }
    let out = std::path::Path::new(&var("OUT_DIR")).join("icon.res");
    let ico = xenon_icon::ico(&xenon_icon::SIZES);
    std::fs::write(&out, xenon_icon::res(&ico)).expect("write icon.res");
    println!("cargo:rustc-link-arg-bins={}", out.display());
}
