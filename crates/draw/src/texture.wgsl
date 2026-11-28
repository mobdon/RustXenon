
// Appended to resolve.wgsl, whose decode bindings (guest, texels, dp) and
// helpers (tiled_b, swap_endian) it uses.

// Tiled DXT blocks from guest bytes to linear BC blocks, as bc_blocks in
// lib.rs does on the CPU: the fetch endian per word, then the 32x32-block
// tiling. One invocation per block; `dp.w` x `dp.h` blocks, `dp.pitch` in
// blocks, `dp.mode` the log2 of the block bytes (3 for BC1, 4 for BC2 and
// BC3), into rows of `dp.row_words` words.
@compute @workgroup_size(8, 8)
fn detile_bc(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= dp.w || id.y >= dp.h {
        return;
    }
    let words = 1u << (dp.mode - 2u);
    let src = dp.offset + (tiled_b(id.x, id.y, dp.pitch, dp.mode) >> 2u);
    let dst = id.y * dp.row_words + id.x * words;
    for (var i = 0u; i < words; i++) {
        texels[dst + i] = swap_endian(guest[src + i], dp.endian);
    }
}
