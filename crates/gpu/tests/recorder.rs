//! Recorder tests on hand-built packet streams (synthetic words only).

use xenon_gpu::pm4::build::{
    type0, type0_one_reg, type1, type2, type3, type3_header, type3_predicated,
};
use xenon_gpu::pm4::*;
use xenon_gpu::regs::*;
use xenon_gpu::*;

const BUF: u32 = 0x0010_0000;
const DATA: u32 = 0x0020_0000;

fn cat(packets: &[Vec<u32>]) -> Vec<u32> {
    packets.concat()
}

/// Memory holding `stream` at BUF.
fn mem_with(stream: &[u32]) -> SliceMemory {
    SliceMemory::from_words(BUF, stream)
}

fn record(stream: &[u32]) -> Recorder {
    let mem = mem_with(stream);
    let mut rec = Recorder::new();
    rec.record_buffer(&mem, BUF, stream.len() as u32).unwrap();
    rec
}

fn non_reg_commands(rec: &Recorder) -> Vec<&Command> {
    rec.commands
        .iter()
        .filter(|c| !matches!(c, Command::WriteRegisters { .. }))
        .collect()
}

fn draw_initiator(prim: u32, source: u32, int32: bool, count: u32) -> u32 {
    prim | (source << 6) | (u32::from(int32) << 11) | (count << 16)
}

#[test]
fn type0_writes_consecutive_registers() {
    let rec = record(&type0(PA_CL_VPORT_XSCALE, &[1, 2, 3]));
    assert_eq!(
        rec.commands,
        vec![Command::WriteRegisters {
            base: PA_CL_VPORT_XSCALE,
            values: vec![1, 2, 3]
        }]
    );
    assert_eq!(rec.regs.get(PA_CL_VPORT_XOFFSET), 2);
    assert_eq!(rec.regs.get(PA_CL_VPORT_YSCALE), 3);
}

#[test]
fn type0_one_reg_writes_same_register() {
    let rec = record(&type0_one_reg(RB_COLOR_MASK, &[0xF, 0xFF]));
    assert_eq!(
        rec.commands,
        vec![
            Command::WriteRegisters {
                base: RB_COLOR_MASK,
                values: vec![0xF]
            },
            Command::WriteRegisters {
                base: RB_COLOR_MASK,
                values: vec![0xFF]
            },
        ]
    );
    assert_eq!(rec.regs.get(RB_COLOR_MASK), 0xFF);
    assert_eq!(rec.regs.get(RB_COLOR_MASK + 1), 0);
}

#[test]
fn type1_writes_two_registers() {
    let rec = record(&type1(SCRATCH_REG0, 0xAA, SCRATCH_ADDR, 0xBB));
    assert_eq!(rec.regs.get(SCRATCH_REG0), 0xAA);
    assert_eq!(rec.regs.get(SCRATCH_ADDR), 0xBB);
    assert_eq!(rec.commands.len(), 2);
}

#[test]
fn type2_and_zero_dwords_are_noops() {
    let rec = record(&cat(&[
        vec![0, 0],
        type2(),
        type3(PM4_NOP, &[0xDEAD, 0xBEEF]),
        type2(),
    ]));
    assert_eq!(rec.commands, vec![Command::Nop, Command::Nop, Command::Nop]);
    assert!(rec.warnings.is_empty());
}

#[test]
fn register_writes_past_the_file_are_dropped() {
    let rec = record(&type0(0x5001, &[1, 2, 3, 4]));
    assert_eq!(
        rec.commands,
        vec![Command::WriteRegisters {
            base: 0x5001,
            values: vec![1, 2]
        }]
    );
    assert_eq!(
        rec.warnings,
        vec![Warning::RegisterOutOfRange { index: 0x5003 }]
    );
}

#[test]
fn ring_wraps_around() {
    // Ring of 16 dwords; a 4-dword packet starts at index 14 and wraps.
    let packet = type0(RB_SURFACE_INFO, &[0x11, 0x22, 0x33]);
    let mut ring = vec![0u32; 16];
    ring[14] = packet[0];
    ring[15] = packet[1];
    ring[0] = packet[2];
    ring[1] = packet[3];
    ring[2..4].copy_from_slice(&type3(PM4_INTERRUPT, &[1]));
    let mem = SliceMemory::from_words(BUF, &ring);
    let mut rec = Recorder::new();
    let new_read = rec.record_ring(&mem, BUF, 16, 14, 4).unwrap();
    assert_eq!(new_read, 4);
    assert_eq!(
        rec.commands,
        vec![
            Command::WriteRegisters {
                base: RB_SURFACE_INFO,
                values: vec![0x11, 0x22, 0x33]
            },
            Command::Interrupt { cpu_mask: 1 },
        ]
    );
}

#[test]
fn ring_without_wrap_and_empty_span() {
    let mut ring = vec![0u32; 8];
    ring[2..4].copy_from_slice(&type3(PM4_INTERRUPT, &[2]));
    let mem = SliceMemory::from_words(BUF, &ring);
    let mut rec = Recorder::new();
    assert_eq!(rec.record_ring(&mem, BUF, 8, 2, 4), Ok(4));
    assert_eq!(rec.record_ring(&mem, BUF, 8, 4, 4), Ok(4));
    assert_eq!(rec.commands, vec![Command::Interrupt { cpu_mask: 2 }]);
}

#[test]
fn ring_base_is_masked_to_physical() {
    let mem = SliceMemory::from_words(BUF, &type3(PM4_INTERRUPT, &[4]));
    let mut rec = Recorder::new();
    rec.record_ring(&mem, 0xA000_0000 | BUF, 8, 0, 2).unwrap();
    assert_eq!(rec.commands, vec![Command::Interrupt { cpu_mask: 4 }]);
}

#[test]
fn ring_errors() {
    let mem = SliceMemory::from_words(BUF, &[0; 4]);
    let mut rec = Recorder::new();
    assert!(matches!(
        rec.record_ring(&mem, BUF, 4, 4, 0),
        Err(Error::RingPointerOutOfRange { .. })
    ));
    assert!(matches!(
        rec.record_ring(&mem, BUF, 0, 0, 0),
        Err(Error::RingPointerOutOfRange { .. })
    ));
    assert!(matches!(
        rec.record_ring(&mem, BUF, 64, 0, 32),
        Err(Error::Unreadable { .. })
    ));
    assert!(matches!(
        rec.record_buffer(&mem, DATA, 4),
        Err(Error::Unreadable { .. })
    ));
}

#[test]
fn indirect_buffers_are_followed_in_order() {
    let ib = type3(PM4_INTERRUPT, &[2]);
    let stream = cat(&[
        type3(PM4_INTERRUPT, &[1]),
        type3(PM4_INDIRECT_BUFFER, &[0x8000_0000 | DATA, ib.len() as u32]),
        type3(PM4_INDIRECT_BUFFER_PFD, &[DATA, ib.len() as u32]),
        type3(PM4_INTERRUPT, &[3]),
    ]);
    let mut mem = mem_with(&stream);
    mem.write_words(DATA, &ib);
    let mut rec = Recorder::new();
    rec.record_buffer(&mem, BUF, stream.len() as u32).unwrap();
    let masks: Vec<u32> = rec
        .commands
        .iter()
        .map(|c| match c {
            Command::Interrupt { cpu_mask } => *cpu_mask,
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(masks, vec![1, 2, 2, 3]);
    assert!(rec.warnings.is_empty());
}

#[test]
fn indirect_depth_limit_stops_self_reference() {
    // An indirect buffer that records an interrupt and then calls itself.
    let ib = cat(&[
        type3(PM4_INTERRUPT, &[1]),
        type3(PM4_INDIRECT_BUFFER, &[DATA, 5]),
    ]);
    assert_eq!(ib.len(), 5);
    let stream = type3(PM4_INDIRECT_BUFFER, &[DATA, 5]);
    let mut mem = mem_with(&stream);
    mem.write_words(DATA, &ib);
    let mut rec = Recorder::new();
    rec.max_indirect_depth = 3;
    rec.record_buffer(&mem, BUF, stream.len() as u32).unwrap();
    assert_eq!(rec.commands.len(), 3);
    assert_eq!(
        rec.warnings,
        vec![Warning::IndirectDepthExceeded {
            addr: DATA,
            dwords: 5
        }]
    );
}

#[test]
fn unreadable_indirect_buffer_warns_and_continues() {
    let stream = cat(&[
        type3(PM4_INDIRECT_BUFFER, &[0x0F00_0000, 4]),
        type3(PM4_INTERRUPT, &[1]),
    ]);
    let rec = record(&stream);
    assert_eq!(
        rec.warnings,
        vec![Warning::Unreadable {
            addr: 0x0F00_0000,
            dwords: 4
        }]
    );
    assert_eq!(rec.commands, vec![Command::Interrupt { cpu_mask: 1 }]);
}

#[test]
fn draw_indx_auto_index() {
    let init = draw_initiator(4, 2, false, 36);
    let rec = record(&type3(PM4_DRAW_INDX, &[0x55, init]));
    assert_eq!(rec.regs.get(VGT_DRAW_INITIATOR), init);
    assert_eq!(
        non_reg_commands(&rec),
        vec![&Command::Draw(Draw {
            primitive: PrimitiveType::TriangleList,
            index_count: 36,
            source: IndexSource::Auto,
            initiator: init,
            viz_query: Some(0x55),
        })]
    );
}

#[test]
fn draw_indx_dma_indices() {
    let init = draw_initiator(6, 0, true, 100);
    let dma_size = (2 << 30) | 100; // 8in32 swap, 100 indices
    let rec = record(&type3(PM4_DRAW_INDX, &[0, init, 0xC030_0006, dma_size]));
    assert_eq!(rec.regs.get(VGT_DMA_BASE), 0xC030_0006);
    assert_eq!(rec.regs.get(VGT_DMA_SIZE), dma_size);
    let Command::Draw(draw) = non_reg_commands(&rec)[0] else {
        panic!()
    };
    assert_eq!(draw.primitive, PrimitiveType::TriangleStrip);
    assert_eq!(
        draw.source,
        IndexSource::Dma {
            address: 0x0030_0004,
            buffer_indices: 100,
            format: IndexFormat::Int32,
            endian: Endian::E8in32
        }
    );
}

#[test]
fn draw_indx_2_auto_and_immediate() {
    let auto = draw_initiator(8, 2, false, 3);
    let imm = draw_initiator(4, 1, false, 4);
    let rec = record(&cat(&[
        type3(PM4_DRAW_INDX_2, &[auto]),
        type3(PM4_DRAW_INDX_2, &[imm, 0x0000_0001, 0x0002_0003]),
    ]));
    let draws: Vec<&Draw> = rec
        .commands
        .iter()
        .filter_map(|c| match c {
            Command::Draw(d) => Some(d),
            _ => None,
        })
        .collect();
    assert_eq!(draws.len(), 2);
    assert_eq!(draws[0].primitive, PrimitiveType::RectangleList);
    assert_eq!(draws[0].source, IndexSource::Auto);
    assert_eq!(draws[0].viz_query, None);
    assert_eq!(
        draws[1].source,
        IndexSource::Immediate {
            format: IndexFormat::Int16,
            words: vec![1, 0x0002_0003]
        }
    );
    assert_eq!(draws[1].index_count, 4);
}

#[test]
fn draw_with_bad_source_or_short_dma_warns() {
    let bad = draw_initiator(4, 3, false, 3);
    let dma = draw_initiator(4, 0, false, 3);
    let rec = record(&cat(&[
        type3(PM4_DRAW_INDX_2, &[bad]),
        type3(PM4_DRAW_INDX_2, &[dma, 0x1000]),
    ]));
    assert!(non_reg_commands(&rec).is_empty());
    assert_eq!(
        rec.warnings,
        vec![
            Warning::BadIndexSource { initiator: bad },
            Warning::PacketTooShort {
                opcode: PM4_DRAW_INDX_2,
                count: 2
            },
        ]
    );
}

#[test]
fn draw_in_copy_mode_is_a_resolve() {
    let copy_control = 1 | (2 << 4) | (1 << 8) | (1 << 20); // RT1, sample 2, color clear, convert
    let dest_info = 2 | (1 << 3) | (3 << 4) | (6 << 7) | (0x3E << 16) | (1 << 24);
    let stream = cat(&[
        type0(
            RB_SURFACE_INFO,
            &[1280 | (2 << 16), 0, 0x0001_0010, 0x0006_0123],
        ),
        type0(
            PA_SC_WINDOW_OFFSET,
            &[
                (0x7FFF << 16) | 8,
                (1 << 31) | (16 << 16) | 32,
                (720 << 16) | 1280,
            ],
        ),
        type0(RB_MODECONTROL, &[EDRAM_MODE_COPY]),
        type0(
            RB_COPY_CONTROL,
            &[copy_control, 0x0040_0000, (720 << 16) | 1280, dest_info],
        ),
        type0(RB_DEPTH_CLEAR, &[0xFFFF_FF00, 0x8080_8080, 0x1234]),
        type0(RB_COPY_SURFACE_SLICE, &[7]),
        type0(PA_SU_SC_MODE_CNTL, &[1 << 16]),
        type0(SHADER_CONSTANT_FETCH, &[DATA | 3, (6 << 2) | 2]),
        type3(PM4_DRAW_INDX, &[0, draw_initiator(8, 2, false, 3)]),
    ]);
    let rec = record(&stream);
    let cmds = non_reg_commands(&rec);
    assert_eq!(cmds.len(), 1);
    let Command::Resolve(r) = cmds[0] else {
        panic!("{cmds:?}")
    };
    assert_eq!(r.src_select, 1);
    assert!(!r.is_depth);
    assert_eq!(r.src_edram_base, 0x123);
    assert_eq!(r.src_format, 6);
    assert_eq!(r.sample_select, 2);
    assert!(r.color_clear_enable && !r.depth_clear_enable);
    assert_eq!(r.copy_command, CopyCommand::Convert);
    assert_eq!(r.dest_base, 0x0040_0000);
    assert_eq!((r.dest_pitch, r.dest_height), (1280, 720));
    assert_eq!(r.dest_endian, 2);
    assert!(r.dest_array && r.dest_swap);
    assert_eq!((r.dest_slice, r.dest_format, r.dest_exp_bias), (3, 6, -2));
    assert_eq!((r.surface_pitch, r.msaa_samples), (1280, 2));
    assert_eq!(
        (r.depth_clear, r.color_clear, r.color_clear_lo),
        (0xFFFF_FF00, 0x8080_8080, 0x1234)
    );
    assert_eq!(r.surface_slice, 7);
    assert_eq!((r.window_offset_x, r.window_offset_y), (8, -1));
    assert!(r.vtx_window_offset_enable && r.window_offset_disable);
    assert_eq!(
        r.window_scissor,
        Rect {
            left: 32,
            top: 16,
            right: 1280,
            bottom: 720
        }
    );
    assert_eq!(r.vertex_fetch0.fetch_type, FetchConstantType::Vertex);
    assert_eq!(
        (r.vertex_fetch0.address, r.vertex_fetch0.size_dwords),
        (DATA, 6)
    );
    assert_eq!(r.vertex_fetch0.endian, Endian::E8in32);
}

#[test]
fn depth_resolve_reads_depth_info() {
    let stream = cat(&[
        type0(RB_DEPTH_INFO, &[(1 << 16) | 0x0805]),
        type0(RB_MODECONTROL, &[EDRAM_MODE_COPY]),
        type0(RB_COPY_CONTROL, &[4 | (1 << 9)]),
        type3(PM4_DRAW_INDX_2, &[draw_initiator(8, 2, false, 3)]),
        // Back to color/depth mode: an ordinary draw again.
        type0(RB_MODECONTROL, &[4]),
        type3(PM4_DRAW_INDX_2, &[draw_initiator(8, 2, false, 3)]),
    ]);
    let rec = record(&stream);
    let cmds = non_reg_commands(&rec);
    let Command::Resolve(r) = cmds[0] else {
        panic!()
    };
    assert!(r.is_depth && r.depth_clear_enable);
    assert_eq!((r.src_edram_base, r.src_format), (0x805, 1));
    assert!(matches!(cmds[1], Command::Draw(_)));
}

#[test]
fn set_constant_types_map_to_register_ranges() {
    let stream = cat(&[
        type3(PM4_SET_CONSTANT, &[4, 0x10, 0x11]),
        type3(PM4_SET_CONSTANT, &[(1 << 16) | 6, 0x20]),
        type3(PM4_SET_CONSTANT, &[(2 << 16) | 1, 0x30]),
        type3(PM4_SET_CONSTANT, &[(3 << 16) | 2, 0x40]),
        type3(PM4_SET_CONSTANT, &[(4 << 16) | 0x208, 0x50]),
        type3(PM4_SET_CONSTANT, &[5 << 16, 0x60]),
    ]);
    let rec = record(&stream);
    assert_eq!(rec.regs.get(SHADER_CONSTANT_FLOAT + 4), 0x10);
    assert_eq!(rec.regs.get(SHADER_CONSTANT_FLOAT + 5), 0x11);
    assert_eq!(rec.regs.get(SHADER_CONSTANT_FETCH + 6), 0x20);
    assert_eq!(rec.regs.get(SHADER_CONSTANT_BOOL + 1), 0x30);
    assert_eq!(rec.regs.get(SHADER_CONSTANT_LOOP + 2), 0x40);
    assert_eq!(rec.regs.get(RB_MODECONTROL), 0x50);
    assert_eq!(
        rec.warnings,
        vec![Warning::BadConstantType {
            offset_type: 5 << 16
        }]
    );
    assert_eq!(rec.commands.len(), 5);
}

#[test]
fn set_constant2_and_shader_constants() {
    let stream = cat(&[
        type3(PM4_SET_CONSTANT2, &[0xABCD_0000 | RB_DEPTHCONTROL, 7, 8]),
        type3(PM4_SET_SHADER_CONSTANTS, &[SHADER_CONSTANT_FLOAT + 8, 9]),
    ]);
    let rec = record(&stream);
    assert_eq!(rec.regs.get(RB_DEPTHCONTROL), 7);
    assert_eq!(rec.regs.get(RB_BLENDCONTROL0), 8);
    assert_eq!(rec.regs.get(SHADER_CONSTANT_FLOAT + 8), 9);
}

#[test]
fn load_alu_constant_reads_guest_memory() {
    let stream = cat(&[
        type3(PM4_LOAD_ALU_CONSTANT, &[0x4000_0000 | DATA, 3, 2]),
        type3(PM4_LOAD_ALU_CONSTANT, &[0x0F00_0000, 0, 4]),
    ]);
    let mut mem = mem_with(&stream);
    mem.write_words(DATA, &[0x3F80_0000, 0x4000_0000]);
    let mut rec = Recorder::new();
    rec.record_buffer(&mem, BUF, stream.len() as u32).unwrap();
    assert_eq!(
        rec.commands,
        vec![Command::WriteRegisters {
            base: SHADER_CONSTANT_FLOAT + 3,
            values: vec![0x3F80_0000, 0x4000_0000]
        }]
    );
    assert_eq!(
        rec.warnings,
        vec![Warning::Unreadable {
            addr: 0x0F00_0000,
            dwords: 4
        }]
    );
}

#[test]
fn im_load_reads_shader_from_memory() {
    let stream = type3(PM4_IM_LOAD, &[DATA | 1, 3]);
    let mut mem = mem_with(&stream);
    mem.write_words(DATA, &[0x1111, 0x2222, 0x3333]);
    let mut rec = Recorder::new();
    rec.record_buffer(&mem, BUF, stream.len() as u32).unwrap();
    assert_eq!(
        rec.commands,
        vec![Command::LoadShader {
            stage: ShaderStage::Pixel,
            source_addr: Some(DATA),
            start: 0,
            words: vec![0x1111, 0x2222, 0x3333]
        }]
    );
}

#[test]
fn im_load_immediate_and_bad_variants() {
    let stream = cat(&[
        type3(PM4_IM_LOAD_IMMEDIATE, &[0, 2, 0xAAAA, 0xBBBB]),
        type3(PM4_IM_LOAD_IMMEDIATE, &[1, 5, 0xCCCC]),
        type3(PM4_IM_LOAD_IMMEDIATE, &[2, 0]),
        type3(PM4_IM_LOAD, &[DATA | 2, 1]),
    ]);
    let rec = record(&stream);
    assert_eq!(
        rec.commands,
        vec![Command::LoadShader {
            stage: ShaderStage::Vertex,
            source_addr: None,
            start: 0,
            words: vec![0xAAAA, 0xBBBB]
        }]
    );
    assert_eq!(
        rec.warnings,
        vec![
            Warning::PacketTooShort {
                opcode: PM4_IM_LOAD_IMMEDIATE,
                count: 3
            },
            Warning::BadShaderType { value: 2 },
            Warning::BadShaderType { value: 2 },
        ]
    );
}

#[test]
fn reg_rmw_immediate_and_register_operands() {
    let stream = cat(&[
        type0(SCRATCH_REG0, &[0xFF00_FF00, 0x0F0F_0F0F, 0x0000_0001]),
        // reg0 = (reg0 & 0xFFFF0000) | 0x12
        type3(PM4_REG_RMW, &[SCRATCH_REG0, 0xFFFF_0000, 0x12]),
        // reg1 = (reg1 & reg0) | reg2
        type3(
            PM4_REG_RMW,
            &[
                (3 << 30) | (SCRATCH_REG0 + 1),
                SCRATCH_REG0,
                SCRATCH_REG0 + 2,
            ],
        ),
    ]);
    let rec = record(&stream);
    assert_eq!(rec.regs.get(SCRATCH_REG0), 0xFF00_0012);
    assert_eq!(
        rec.regs.get(SCRATCH_REG0 + 1),
        (0x0F0F_0F0F & 0xFF00_0012) | 1
    );
    assert_eq!(
        rec.commands.last(),
        Some(&Command::WriteRegisters {
            base: SCRATCH_REG0 + 1,
            values: vec![0x0F00_0003]
        })
    );
}

#[test]
fn swap_and_predicated_swap() {
    let stream = cat(&[
        type3(PM4_XE_SWAP, &[SWAP_SIGNATURE, DATA, 1280, 720, 0, 0]),
        type3_predicated(PM4_XE_SWAP, &[SWAP_SIGNATURE, DATA, 1, 1]),
        type3(PM4_XE_SWAP, &[0, 0, 0]),
    ]);
    let rec = record(&stream);
    assert_eq!(
        rec.commands,
        vec![Command::Swap {
            signature_ok: true,
            frontbuffer: DATA,
            width: 1280,
            height: 720
        }]
    );
    assert_eq!(
        rec.warnings,
        vec![Warning::PacketTooShort {
            opcode: PM4_XE_SWAP,
            count: 3
        }]
    );
}

#[test]
fn predication_follows_bin_mask() {
    let stream = cat(&[
        type3_predicated(PM4_INTERRUPT, &[1]),
        type3(PM4_SET_BIN_MASK_LO, &[0]),
        type3_predicated(PM4_INTERRUPT, &[2]),
        type3(PM4_INTERRUPT, &[3]),
        type3(PM4_SET_BIN_MASK, &[1, 0]),
        type3(PM4_SET_BIN_SELECT, &[1, 0]),
        type3_predicated(PM4_INTERRUPT, &[4]),
    ]);
    let rec = record(&stream);
    let masks: Vec<_> = rec
        .commands
        .iter()
        .map(|c| match c {
            Command::Interrupt { cpu_mask } => *cpu_mask,
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(masks, vec![1, 3, 4]);
}

#[test]
fn event_writes_and_initiator_register() {
    let stream = cat(&[
        type3(PM4_EVENT_WRITE, &[0x46]),
        type3(PM4_EVENT_WRITE_SHD, &[0x8000_0003, DATA | 2, 0x77]),
        type3(PM4_EVENT_WRITE_EXT, &[0x19, DATA | 1]),
        type3(PM4_EVENT_WRITE_ZPD, &[0x15]),
    ]);
    let rec = record(&stream);
    assert_eq!(rec.regs.get(VGT_EVENT_INITIATOR), 0x15);
    let events: Vec<EventWrite> = rec
        .commands
        .iter()
        .filter_map(|c| match c {
            Command::EventWrite(e) => Some(e.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(events.len(), 4);
    assert_eq!(events[0].kind, EventKind::Plain);
    assert_eq!(events[0].address, None);
    assert_eq!(
        events[1],
        EventWrite {
            kind: EventKind::Shader,
            initiator: 0x8000_0003,
            address: Some(DATA),
            endian: Endian::E8in32,
            value: Some(0x77),
            use_counter: true,
        }
    );
    assert_eq!(
        (events[2].kind, events[2].address, events[2].endian),
        (EventKind::ScreenExtent, Some(DATA), Endian::E8in16)
    );
    assert_eq!(events[3].kind, EventKind::ZPassDone);
    // The initiator write is recorded before each event.
    assert_eq!(
        rec.commands[0],
        Command::WriteRegisters {
            base: VGT_EVENT_INITIATOR,
            values: vec![6]
        }
    );
}

#[test]
fn memory_writes_are_recorded_not_applied() {
    let stream = cat(&[
        type3(PM4_MEM_WRITE, &[0xA000_0000 | DATA | 2, 1, 2]),
        type3(PM4_REG_TO_MEM, &[SCRATCH_REG0, DATA | 1]),
    ]);
    let mem = mem_with(&stream);
    let mut rec = Recorder::new();
    rec.record_buffer(&mem, BUF, stream.len() as u32).unwrap();
    assert_eq!(
        rec.commands,
        vec![
            Command::MemWrite {
                addr: DATA,
                endian: Endian::E8in32,
                values: vec![1, 2]
            },
            Command::RegToMem {
                register: SCRATCH_REG0,
                addr: DATA,
                endian: Endian::E8in16
            },
        ]
    );
    assert_eq!(mem.read_bytes(DATA, 4), None);
}

#[test]
fn wait_reg_mem_and_cond_write() {
    let stream = cat(&[
        type3(PM4_WAIT_REG_MEM, &[0x13, DATA | 2, 5, 0xFF, 0x100]),
        type3(
            PM4_WAIT_REG_MEM,
            &[0x06, COHER_STATUS_HOST, 0, 0x8000_0000, 10],
        ),
        type3(
            PM4_COND_WRITE,
            &[0x103, SCRATCH_REG0, 1, 0xFFFF_FFFF, DATA | 2, 9],
        ),
    ]);
    let rec = record(&stream);
    let Command::WaitRegMem(w) = rec.commands[0] else {
        panic!()
    };
    assert_eq!(
        w,
        WaitRegMem {
            function: 3,
            memory: true,
            poll_addr: DATA,
            endian: Endian::E8in32,
            reference: 5,
            mask: 0xFF,
            wait: 0x100
        }
    );
    assert!(w.matches(0x105) && !w.matches(4));
    let Command::WaitRegMem(w) = rec.commands[1] else {
        panic!()
    };
    assert!(!w.memory && w.poll_addr == COHER_STATUS_HOST && w.function == 6);
    assert!(w.matches(0x8000_0000) && !w.matches(0x7FFF_FFFF));
    let Command::CondWrite {
        wait,
        write_memory,
        write_addr,
        write_endian,
        write_data,
    } = rec.commands[2]
    else {
        panic!()
    };
    assert!(write_memory && !wait.memory);
    assert_eq!(
        (write_addr, write_endian, write_data),
        (DATA, Endian::E8in32, 9)
    );
}

#[test]
fn misc_opcodes() {
    let stream = cat(&[
        type3(PM4_ME_INIT, &[1, 2, 3]),
        type3(PM4_CONTEXT_UPDATE, &[0]),
        type3(PM4_INVALIDATE_STATE, &[0x7FFF]),
        type3(PM4_WAIT_FOR_IDLE, &[0]),
        type3(PM4_VIZ_QUERY, &[5]),
        type3(PM4_VIZ_QUERY, &[0x100 | 33]),
        type3(PM4_DRAW_INDX_BIN, &[9, 8]),
    ]);
    let rec = record(&stream);
    assert_eq!(
        non_reg_commands(&rec),
        vec![
            &Command::MeInit {
                words: vec![1, 2, 3]
            },
            &Command::ContextUpdate { value: 0 },
            &Command::InvalidateState { mask: 0x7FFF },
            &Command::WaitForIdle { value: 0 },
            &Command::VizQuery { id: 5, end: false },
            &Command::VizQuery { id: 33, end: true },
            &Command::Unknown {
                opcode: PM4_DRAW_INDX_BIN,
                words: vec![9, 8]
            },
        ]
    );
    assert_eq!(rec.regs.get(PA_SC_VIZ_QUERY_STATUS_1), 2);
    assert_eq!(rec.regs.get(VGT_EVENT_INITIATOR), 8);
}

#[test]
fn truncated_packet_warns_and_stops_buffer() {
    let mut stream = type3(PM4_INTERRUPT, &[1]);
    stream.push(type3_header(PM4_ME_INIT, 10, false));
    stream.extend([1, 2]);
    let rec = record(&stream);
    assert_eq!(rec.commands, vec![Command::Interrupt { cpu_mask: 1 }]);
    assert_eq!(
        rec.warnings,
        vec![Warning::Truncated {
            header: stream[2],
            available: 2
        }]
    );
}

#[test]
fn short_packet_is_skipped_and_stream_continues() {
    let stream = cat(&[
        type3(PM4_WAIT_REG_MEM, &[3]),
        type3(PM4_REG_RMW, &[1, 2]),
        type3(PM4_INDIRECT_BUFFER, &[DATA]),
        type3(PM4_INTERRUPT, &[1]),
    ]);
    let rec = record(&stream);
    assert_eq!(rec.commands, vec![Command::Interrupt { cpu_mask: 1 }]);
    assert_eq!(rec.warnings.len(), 3);
    assert!(rec
        .warnings
        .iter()
        .all(|w| matches!(w, Warning::PacketTooShort { .. })));
}

#[test]
fn garbage_streams_never_panic() {
    // Deterministic pseudo-random words, biased towards type-3 headers.
    let mut state = 0x1234_5678u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for _ in 0..200 {
        let words: Vec<u32> = (0..64)
            .map(|i| {
                let w = next();
                if i % 3 == 0 {
                    (w | 0xC000_0000) & 0xC00F_FFFF
                } else {
                    w
                }
            })
            .collect();
        let mut mem = SliceMemory::from_words(0, &words);
        mem.write_words(0x1000, &words);
        let mut rec = Recorder::new();
        rec.record_buffer(&mem, 0, words.len() as u32).unwrap();
        let _ = rec.record_ring(&mem, 0, 64, 40, 10);
    }
}

#[test]
fn fetch_constants_from_registers() {
    let words = [DATA | 3, (0x30 << 2) | 1, 0xA, 0xB, 0xC, 0xD];
    let rec = record(&type3(
        PM4_SET_CONSTANT,
        &[
            (1 << 16) | 6,
            words[0],
            words[1],
            words[2],
            words[3],
            words[4],
            words[5],
        ],
    ));
    assert_eq!(rec.regs.fetch_constant(1), words);
    let vf = rec.regs.vertex_fetch(3);
    assert_eq!(vf, VertexFetch::decode([words[0], words[1]]));
    assert_eq!(
        vf,
        VertexFetch {
            fetch_type: FetchConstantType::Vertex,
            address: DATA,
            endian: Endian::E8in16,
            size_dwords: 0x30
        }
    );
}

#[test]
fn replaying_register_writes_rebuilds_the_file() {
    let stream = cat(&[
        type0(RB_SURFACE_INFO, &[1, 2, 3]),
        type1(SCRATCH_REG0, 5, SCRATCH_UMSK, 6),
        type3(PM4_SET_CONSTANT, &[2, 7, 8]),
        type3(PM4_REG_RMW, &[SCRATCH_REG0, 0, 0x99]),
        type3(PM4_DRAW_INDX, &[0, draw_initiator(4, 0, false, 3), DATA, 3]),
        type3(PM4_EVENT_WRITE, &[4]),
    ]);
    let rec = record(&stream);
    let mut replay = RegisterFile::default();
    for c in &rec.commands {
        if let Command::WriteRegisters { base, values } = c {
            for (i, v) in values.iter().enumerate() {
                replay.set(base + i as u32, *v);
            }
        }
    }
    for reg in [
        RB_SURFACE_INFO,
        RB_COLOR_INFO,
        RB_DEPTH_INFO,
        SCRATCH_REG0,
        SCRATCH_UMSK,
        VGT_DRAW_INITIATOR,
        VGT_DMA_BASE,
        VGT_DMA_SIZE,
        VGT_EVENT_INITIATOR,
        SHADER_CONSTANT_FLOAT + 2,
        SHADER_CONSTANT_FLOAT + 3,
    ] {
        assert_eq!(replay.get(reg), rec.regs.get(reg), "register {reg:#X}");
    }
    assert_eq!(rec.regs.get(SCRATCH_REG0), 0x99);
}

#[test]
fn huge_ring_pointers_do_not_overflow() {
    // read_ptr * 4 overflows u32; this panicked in debug builds before.
    let mem = SliceMemory::from_words(BUF, &[0; 4]);
    let mut rec = Recorder::new();
    for (size, read, write) in [
        (u32::MAX, 0x8000_0000, 0x8000_0004),
        (u32::MAX, 0xFFFF_FFF0, 2),
        (0x4000_0001, 0x4000_0000, 0x4000_0000),
    ] {
        let _ = rec.record_ring(&mem, BUF, size, read, write);
    }
    assert!(rec.commands.is_empty());
}

#[test]
fn out_of_range_fetch_constant_reads_zero() {
    let rec = Recorder::new();
    assert_eq!(rec.regs.fetch_constant(u32::MAX), [0; 6]);
    assert_eq!(rec.regs.vertex_fetch(u32::MAX).address, 0);
}
