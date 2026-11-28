//! Presents frames through a swapchain on the drawer's own device: the
//! drawer's frame texture is scaled into the window on the GPU, with no
//! read-back and no CPU copy, and shown at the monitor's refresh (see
//! [`Presenter::follow_refresh`]).

use std::sync::Arc;
use winit::window::Window;

/// The drawer's device, its adapter and queue, and the instance they came
/// from (a surface must come from the same instance).
pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// Scales a texture into the letterbox rectangle the viewport sets,
/// bilinear and clamped at the edges, as [`crate::blit`] does on the CPU.
const SHADER: &str = r"
@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;

struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> Out {
    // One triangle over the viewport: uv 0..1 inside it.
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: Out;
    o.pos = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    o.uv = uv;
    return o;
}

@fragment
fn fs(v: Out) -> @location(0) vec4<f32> {
    return vec4<f32>(textureSample(frame, linear_sampler, v.uv).rgb, 1.0);
}
";

pub struct Presenter {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    /// Where CPU frames and the text screen are uploaded to, kept for the
    /// next of the same size.
    upload: Option<wgpu::Texture>,
    /// Set when the swapchain no longer matches the window.
    stale: bool,
    /// The present modes the surface offers.
    modes: Vec<wgpu::PresentMode>,
}

impl Presenter {
    pub fn new(gpu: Gpu, window: Arc<Window>) -> Result<Presenter, String> {
        let Gpu {
            instance,
            adapter,
            device,
            queue,
        } = gpu;
        let size = window.inner_size();
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::from(window))
            .map_err(|e| e.to_string())?;
        if !adapter.is_surface_supported(&surface) {
            return Err("the drawer's adapter cannot present to this window".into());
        }
        let caps = surface.get_capabilities(&adapter);
        let mut config =
            (surface.get_default_config(&adapter, size.width.max(1), size.height.max(1)))
                .ok_or("no surface format")?;
        // Frame bytes are shown as they are, as the CPU path shows them:
        // a format that does not convert from sRGB on the way out.
        config.format = (caps.formats.iter().copied())
            .find(|f| !f.is_srgb())
            .unwrap_or(config.format);
        let modes = caps.present_modes.clone();
        config.desired_maximum_frame_latency = 2;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        surface.configure(&device, &config);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("present"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("present"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(config.format.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("present"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(format!("{e:?}"));
        }
        let mut p = Presenter {
            surface,
            device,
            queue,
            config,
            pipeline,
            sampler,
            upload: None,
            stale: false,
            modes,
        };
        p.follow_refresh(0, false);
        eprintln!(
            "window: frames presented on the GPU ({:?}, {:?})",
            p.config.format, p.config.present_mode
        );
        Ok(p)
    }

    /// Picks the present mode for a monitor refreshing at `mhz` millihertz
    /// (0 for not known). At about 60 Hz the host raises the guest's
    /// vertical blank on the display's own (see `xenon_window::vsync`), so
    /// a frame comes each refresh and Fifo shows each once, in order. At
    /// other rates the guest keeps its own 60 Hz clock, and Mailbox shows
    /// the newest frame each refresh without a present ever waiting (Fifo
    /// there blocked the window to about 42 of 60 frames a second).
    /// XENON_PRESENT=fifo, mailbox or immediate picks one; an unsupported
    /// choice falls back.
    pub fn follow_refresh(&mut self, mhz: u32, locked: bool) {
        let want = match std::env::var("XENON_PRESENT").as_deref() {
            Ok("fifo") => wgpu::PresentMode::Fifo,
            Ok("immediate") => wgpu::PresentMode::Immediate,
            Ok("mailbox") => wgpu::PresentMode::Mailbox,
            // Fifo against a title on a clock of its own blocked the
            // window to about 42 of 60 frames a second.
            _ if locked && crate::pace::near_sixty_hz(mhz) => wgpu::PresentMode::Fifo,
            _ => wgpu::PresentMode::Mailbox,
        };
        let mode = [want, wgpu::PresentMode::Mailbox, wgpu::PresentMode::Fifo]
            .into_iter()
            .find(|m| self.modes.contains(m))
            .unwrap_or(wgpu::PresentMode::Fifo);
        if mode != self.config.present_mode {
            self.config.present_mode = mode;
            self.stale = true;
            eprintln!(
                "window: frames presented on the GPU ({:?}, {:?}, monitor {} mHz)",
                self.config.format, mode, mhz
            );
        }
    }

    /// `rgba8` (`width` x `height`, rows top to bottom) in a texture to
    /// present.
    pub fn upload(&mut self, width: u32, height: u32, rgba8: &[u8]) -> wgpu::Texture {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        // Errors here are the window's, not the drawer's.
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let tex = match self.upload.take() {
            Some(t) if t.size() == size => t,
            _ => self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("uploaded frame"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }),
        };
        self.queue.write_texture(
            tex.as_image_copy(),
            rgba8,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            size,
        );
        if let Some(e) = pollster::block_on(scope.pop()) {
            eprintln!("window: upload: {e}");
        }
        self.upload = Some(tex.clone());
        tex
    }

    /// Shows `tex` letterboxed in a `w` x `h` window; false when the window
    /// cannot show it now (hidden, or its swapchain is being remade).
    pub fn present(&mut self, tex: &wgpu::Texture, w: u32, h: u32) -> Result<bool, String> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        if self.stale || (w, h) != (self.config.width, self.config.height) {
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.device, &self.config);
            self.stale = false;
        }
        use wgpu::CurrentSurfaceTexture as Got;
        let out = match self.surface.get_current_texture() {
            Got::Success(t) => t,
            Got::Suboptimal(t) => {
                self.stale = true;
                t
            }
            Got::Timeout | Got::Occluded => return Ok(false),
            Got::Outdated | Got::Lost => {
                self.stale = true;
                return Ok(false);
            }
            Got::Validation => {
                let e = pollster::block_on(scope.pop());
                return Err(e.map_or("validation error".into(), |e| e.to_string()));
            }
        };
        let view = out.texture.create_view(&Default::default());
        let src = tex.create_view(&Default::default());
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("present"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            if let Some((x, y, fw, fh)) = crate::fit(tex.width(), tex.height(), w, h) {
                pass.set_viewport(x as f32, y as f32, fw as f32, fh as f32, 0.0, 1.0);
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        self.queue.submit([enc.finish()]);
        self.queue.present(out);
        match pollster::block_on(scope.pop()) {
            Some(e) => Err(e.to_string()),
            None => Ok(true),
        }
    }
}
