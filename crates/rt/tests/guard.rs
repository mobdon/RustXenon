//! Guarded guest memory, in a process of its own (one guarded memory per
//! process). The wild-access case runs this test binary again as a child.

use xenon_rt::{guard, GuestMemory};

fn wild_child() -> ! {
    let m = GuestMemory::new_guarded(1 << 32);
    guard::on_wild_access(|a| {
        eprintln!("wild guest access at {a:#010x}");
        std::process::exit(3)
    });
    guard::commit(0x8200_0000, 0x1000);
    m.write_u32(0x8200_0000, 1);
    // Never committed: must fault into the hook, not read zero.
    let v = m.read_u32(0x1234_5678);
    eprintln!("read {v:#x} without a fault");
    std::process::exit(0)
}

#[test]
fn guarded_memory_commits_on_demand_and_stops_on_wild_access() {
    if std::env::var_os("XENON_GUARD_CHILD").is_some() {
        wild_child();
    }
    let m = GuestMemory::new_guarded(1 << 32);
    assert!(guard::active());
    assert!(!guard::readable(0x8200_0000, 4));
    guard::commit(0x8200_0000, 0x2000);
    assert!(guard::readable(0x8200_0000, 0x2000));
    assert!(
        !guard::readable(0x8200_1ffe, 4),
        "the third page stays reserved"
    );
    m.write_u32(0x8200_1ffc, 0xDEAD_BEEF);
    assert_eq!(m.read_u32(0x8200_1ffc), 0xDEAD_BEEF);
    assert_eq!(m.read_u32(0x8200_0000), 0, "committed pages start zeroed");
    // Committing again is a no-op; the mirrored views fold onto 0xA0000000.
    guard::commit(0x8200_0000, 0x2000);
    guard::commit(0xA000_1000, 0x1000);
    assert!(guard::readable(0xE000_0000, 0x1000));
    m.write_u32(0xA000_1000, 7);
    assert_eq!(m.read_u32(0xE000_0000), 7);
    assert_eq!(guard::guest_address(0), None);
    drop(m);
    assert!(!guard::active());

    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "guarded_memory_commits_on_demand_and_stops_on_wild_access",
            "--nocapture",
        ])
        .env("XENON_GUARD_CHILD", "1")
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("wild guest access at 0x12345678"), "{err}");
    assert_eq!(out.status.code(), Some(3), "{err}");
}
