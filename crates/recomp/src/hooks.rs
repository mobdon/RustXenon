//! The per-game file, `games/<sha256 of the executable>/hooks.toml`.
//!
//! Holds only what our own analysis needs help with and the hook roles, by
//! guest address. Format is documented in `games/README.md`.

use crate::xex::{err, Error, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hooks {
    /// SHA-256 of the executable, lowercase hex; checked when present.
    pub sha256: Option<String>,
    /// Role name to guest function address.
    #[serde(default)]
    pub hooks: BTreeMap<String, u32>,
    #[serde(default)]
    pub functions: Functions,
    #[serde(default)]
    pub jump_tables: Vec<JumpTableOverride>,
    /// Ranges inside code sections that hold data, not instructions.
    #[serde(default)]
    pub not_code: Vec<Range>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Functions {
    /// Function starts analysis missed.
    #[serde(default)]
    pub extra: Vec<u32>,
    /// Functions whose end analysis gets wrong: `[start, end)`.
    #[serde(default)]
    pub bounds: Vec<Range>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Range {
    pub start: u32,
    pub end: u32,
}

impl Range {
    pub fn contains(&self, a: u32) -> bool {
        a >= self.start && a < self.end
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JumpTableOverride {
    /// Address of the `bctr` that uses the table.
    pub bctr: u32,
    pub targets: Vec<u32>,
    pub default: Option<u32>,
}

impl Hooks {
    /// Parses a hooks file for the executable with SHA-256 `sha256`.
    pub fn parse(text: &str, sha256: &str) -> Result<Hooks> {
        let h: Hooks = toml::from_str(text).map_err(|e| Error(format!("hooks.toml: {e}")))?;
        if let Some(s) = &h.sha256 {
            if !s.eq_ignore_ascii_case(sha256) {
                return err(format!("hooks.toml is for executable {s}, not {sha256}"));
            }
        }
        for r in h.functions.bounds.iter().chain(&h.not_code) {
            if r.end <= r.start {
                return err(format!(
                    "hooks.toml: empty range {:#x}..{:#x}",
                    r.start, r.end
                ));
            }
        }
        Ok(h)
    }

    /// Where the hooks file for `sha256` lives under `games_dir`.
    pub fn path(games_dir: &Path, sha256: &str) -> PathBuf {
        games_dir.join(sha256).join("hooks.toml")
    }

    pub fn is_data(&self, addr: u32) -> bool {
        self.not_code.iter().any(|r| r.contains(addr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_documented_example_and_checks_hash() {
        let text = r#"
            sha256 = "ABCD"
            [hooks]
            main_loop_tick = 0x82000000
            [functions]
            extra = [0x82000300]
            bounds = [{ start = 0x82000400, end = 0x82000440 }]
            [[jump_tables]]
            bctr = 0x82000500
            targets = [0x82000510, 0x82000520]
            default = 0x82000530
            [[not_code]]
            start = 0x82000600
            end = 0x82000700
        "#;
        let h = Hooks::parse(text, "abcd").unwrap();
        assert_eq!(h.hooks["main_loop_tick"], 0x8200_0000);
        assert_eq!(h.functions.extra, [0x8200_0300]);
        assert_eq!(h.jump_tables[0].targets.len(), 2);
        assert!(h.is_data(0x8200_0600) && !h.is_data(0x8200_0700));
        assert!(Hooks::parse(text, "ffff").is_err());
        assert!(Hooks::parse("typo = 1", "").is_err());
        assert!(Hooks::parse("[[not_code]]\nstart = 2\nend = 1", "").is_err());
    }
}
