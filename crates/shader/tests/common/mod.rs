#![allow(dead_code)]

use xenon_shader::{translate, ShaderStage, Translated};

/// Parses and validates WGSL with naga, panicking with the source on error.
pub fn validate(wgsl: &str) {
    let module = match naga::front::wgsl::parse_str(wgsl) {
        Ok(m) => m,
        Err(e) => panic!("WGSL parse error:\n{}\n---\n{wgsl}", e.emit_to_string(wgsl)),
    };
    if let Err(e) = naga::valid::Validator::new(
        naga::valid::ValidationFlags::default(),
        naga::valid::Capabilities::default(),
    )
    .validate(&module)
    {
        panic!(
            "WGSL validation error:\n{}\n---\n{wgsl}",
            e.emit_to_string(wgsl)
        );
    }
}

/// Translates, validates and returns the result.
pub fn ok(words: &[u32], stage: ShaderStage) -> Translated {
    let t = translate(words, stage).unwrap_or_else(|e| panic!("translate failed: {e}"));
    validate(&t.wgsl);
    t
}

/// Asserts that the generated source contains `needle` (whitespace-trimmed
/// line fragment).
pub fn has(t: &Translated, needle: &str) {
    assert!(
        t.wgsl.contains(needle),
        "missing `{needle}` in:\n{}",
        t.wgsl
    );
}

/// Small deterministic generator for fuzz-style tests.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}
