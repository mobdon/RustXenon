//! Test double for `xenon-regress`: follows the program contract with
//! synthetic frames only (no game). Used by the runner's own tests.
//!
//! Extra options: `--noise N` changes N pixels a lot, `--broken` draws a
//! wrong square, `--crash` exits with status 3, `--hang` never exits,
//! `--skip N` does not write frame N.

use std::path::PathBuf;
use xenon_compare::{save_png, Image};

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut frames, mut dir) = (Vec::new(), None);
    let (mut noise, mut broken, mut skip) = (0u32, false, None);
    while let Some(a) = args.next() {
        let mut val = || args.next().expect("option needs a value");
        match a.as_str() {
            "--capture-frames" => {
                frames = val()
                    .split(',')
                    .map(|f| f.parse::<u64>().unwrap())
                    .collect()
            }
            "--capture-dir" => dir = Some(PathBuf::from(val())),
            "--press" => println!("input {}", val()),
            "--noise" => noise = val().parse().unwrap(),
            "--skip" => skip = Some(val().parse::<u64>().unwrap()),
            "--broken" => broken = true,
            "--crash" => std::process::exit(3),
            "--hang" => loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
            },
            _ => println!("ignored {a}"),
        }
    }
    let dir = dir.expect("--capture-dir missing");
    for f in frames {
        if skip == Some(f) {
            continue;
        }
        let mut img = Image::filled(64, 36, [0, 0, 0, 255]);
        for y in 0..36 {
            for x in 0..64 {
                img.set_pixel(x, y, [(x * 4) as u8, (y * 7) as u8, (f % 256) as u8, 255]);
            }
        }
        for i in 0..noise {
            img.set_pixel(i % 64, i / 64 % 36, [255, 255, 255, 255]);
        }
        if broken {
            for y in 10..30 {
                for x in 20..50 {
                    img.set_pixel(x, y, [255, 0, 255, 255]);
                }
            }
        }
        save_png(&dir.join(format!("frame-{f}.png")), &img).unwrap();
    }
}
