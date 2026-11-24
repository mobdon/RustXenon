//! Decoding through FFmpeg, loaded at run time.
//!
//! FFmpeg is LGPL; we use it on the conditions in CREDITS.md. Nothing of it
//! is linked into our binaries: [`Ffmpeg::load`] opens the user's
//! `avutil` and `avcodec` shared libraries (any major version from 58,
//! FFmpeg 4, to 62) and refuses one whose licence is not LGPL, so a GPL or
//! non-free build is never used. Set `XENON_FFMPEG_DIR` to the folder that
//! holds them, or leave it unset to use the system's search path.
//! `tools/ffmpeg-lgpl/build.sh` builds a small matching one.
//!
//! Only a few public structure fields are touched (their offsets have not
//! moved since FFmpeg 4); everything else goes through functions and
//! AVOptions.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fmt;
use std::path::PathBuf;

use libloading::Library;

type P = *mut c_void;

#[derive(Debug)]
pub enum Error {
    /// No loadable FFmpeg was found; holds the names tried.
    NotFound(String),
    /// The FFmpeg found is not an LGPL build; holds its licence string.
    NotLgpl(String),
    /// The library lacks a function we need.
    Missing(String),
    /// No decoder of that name in this FFmpeg build.
    NoDecoder(String),
    /// An FFmpeg call failed: the call and its error code.
    Call(&'static str, i32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::NotFound(t) => write!(f, "no FFmpeg found (tried {t}); set XENON_FFMPEG_DIR"),
            Error::NotLgpl(l) => write!(
                f,
                "refusing FFmpeg build licensed \"{l}\": an LGPL-only build is required"
            ),
            Error::Missing(s) => write!(f, "FFmpeg lacks {s}"),
            Error::NoDecoder(n) => write!(f, "this FFmpeg build has no \"{n}\" decoder"),
            Error::Call(c, e) => write!(f, "{c} failed ({e})"),
        }
    }
}

impl std::error::Error for Error {}

const EAGAIN: c_int = if cfg!(target_os = "macos") { -35 } else { -11 };
const EOF: c_int = -(0x2046_4F45); // FFERRTAG('E','O','F',' ')
const AVMEDIA_TYPE_AUDIO: c_int = 1;
const PADDING: usize = 64; // AV_INPUT_BUFFER_PADDING_SIZE

// Field offsets on 64-bit targets, unchanged from FFmpeg 4 to 7.
const CODEC_ID: usize = 20; // AVCodec.id
const PAR_TYPE: usize = 0; // AVCodecParameters.codec_type
const PAR_ID: usize = 4; //   .codec_id
const PAR_EXTRADATA: usize = 16; // .extradata
const PAR_EXTRADATA_SIZE: usize = 24; // .extradata_size
const PKT_DATA: usize = 24; // AVPacket.data
const FRAME_EXTENDED_DATA: usize = 96; // AVFrame.extended_data
const FRAME_NB_SAMPLES: usize = 112; // AVFrame.nb_samples
const FRAME_FORMAT: usize = 116; // AVFrame.format

unsafe fn field<T: Copy>(p: P, off: usize) -> T {
    (p as *const u8).add(off).cast::<T>().read_unaligned()
}
unsafe fn set_field<T>(p: P, off: usize, v: T) {
    (p as *mut u8).add(off).cast::<T>().write_unaligned(v)
}

macro_rules! api {
    ($($lib:ident $name:ident : fn($($a:ty),*) $(-> $r:ty)?;)*) => {
        /// A loaded LGPL FFmpeg: the libraries and the functions we call.
        pub struct Ffmpeg {
            $($name: unsafe extern "C" fn($($a),*) $(-> $r)?,)*
            license: String,
            version: u32,
            _avcodec: Library,
            _avutil: Library,
        }

        impl Ffmpeg {
            fn bind(avutil: Library, avcodec: Library) -> Result<Self, Error> {
                $(
                    let $name = unsafe {
                        *$lib(&avutil, &avcodec)
                            .get::<unsafe extern "C" fn($($a),*) $(-> $r)?>(concat!(stringify!($name), "\0").as_bytes())
                            .map_err(|_| Error::Missing(stringify!($name).into()))?
                    };
                )*
                let mut ff = Self { $($name,)* license: String::new(), version: 0, _avcodec: avcodec, _avutil: avutil };
                ff.license = unsafe { CStr::from_ptr((ff.avcodec_license)()) }.to_string_lossy().into_owned();
                ff.version = unsafe { (ff.avcodec_version)() };
                Ok(ff)
            }
        }
    };
}

fn u<'a>(u: &'a Library, _: &'a Library) -> &'a Library {
    u
}
fn c<'a>(_: &'a Library, c: &'a Library) -> &'a Library {
    c
}

api! {
    c avcodec_version: fn() -> u32;
    c avcodec_license: fn() -> *const c_char;
    c avcodec_find_decoder_by_name: fn(*const c_char) -> P;
    c avcodec_alloc_context3: fn(P) -> P;
    c avcodec_free_context: fn(*mut P);
    c avcodec_open2: fn(P, P, *mut P) -> c_int;
    c avcodec_parameters_alloc: fn() -> P;
    c avcodec_parameters_free: fn(*mut P);
    c avcodec_parameters_to_context: fn(P, P) -> c_int;
    c avcodec_send_packet: fn(P, P) -> c_int;
    c avcodec_receive_frame: fn(P, P) -> c_int;
    c av_packet_alloc: fn() -> P;
    c av_packet_free: fn(*mut P);
    c av_new_packet: fn(P, c_int) -> c_int;
    c av_packet_unref: fn(P);
    u av_frame_alloc: fn() -> P;
    u av_frame_free: fn(*mut P);
    u av_frame_unref: fn(P);
    u av_opt_set: fn(P, *const c_char, *const c_char, c_int) -> c_int;
    u av_opt_set_int: fn(P, *const c_char, i64, c_int) -> c_int;
    u av_mallocz: fn(usize) -> P;
}

fn lib_names(base: &str, major: u32) -> String {
    if cfg!(windows) {
        format!("{base}-{major}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{base}.{major}.dylib")
    } else {
        format!("lib{base}.so.{major}")
    }
}

impl Ffmpeg {
    /// Finds and opens FFmpeg (see the module docs). Fails with
    /// [`Error::NotLgpl`] for a GPL or non-free build.
    pub fn load() -> Result<Self, Error> {
        if usize::BITS != 64 {
            return Err(Error::Missing("a 64-bit target".into()));
        }
        let dir = std::env::var_os("XENON_FFMPEG_DIR").map(PathBuf::from);
        let path = |name: String| match &dir {
            Some(d) => d.join(name).into_os_string(),
            None => name.into(),
        };
        let mut tried = Vec::new();
        // avcodec major N ships with avutil major N - 2. avutil goes first so
        // avcodec's dependency on it resolves to the same file.
        for major in (58..=62).rev() {
            let (pu, pc) = (
                path(lib_names("avutil", major - 2)),
                path(lib_names("avcodec", major)),
            );
            tried.push(pc.to_string_lossy().into_owned());
            // SAFETY: loading FFmpeg runs only its own library constructors.
            let Ok(avutil) = (unsafe { Library::new(&pu) }) else {
                continue;
            };
            let Ok(avcodec) = (unsafe { Library::new(&pc) }) else {
                continue;
            };
            let ff = Self::bind(avutil, avcodec)?;
            if !ff.license.starts_with("LGPL") {
                return Err(Error::NotLgpl(ff.license));
            }
            return Ok(ff);
        }
        Err(Error::NotFound(tried.join(", ")))
    }

    /// The build's licence string, e.g. "LGPL version 2.1 or later".
    pub fn license(&self) -> &str {
        &self.license
    }

    /// libavcodec's version as major.minor.micro.
    pub fn version(&self) -> (u32, u32, u32) {
        (
            self.version >> 16,
            (self.version >> 8) & 0xff,
            self.version & 0xff,
        )
    }
}

/// One open audio decoder. Output is interleaved f32 in -1..=1.
pub struct Decoder<'a> {
    ff: &'a Ffmpeg,
    ctx: P,
    pkt: P,
    frame: P,
    channels: usize,
}

// SAFETY: the decoder's FFmpeg objects are owned by it alone and FFmpeg
// allows a codec context to move between threads between calls.
unsafe impl Send for Decoder<'_> {}

impl<'a> Decoder<'a> {
    /// Opens the FFmpeg decoder `codec` (e.g. "xma2", "pcm_s16be").
    pub fn open(
        ff: &'a Ffmpeg,
        codec: &str,
        sample_rate: u32,
        channels: u16,
        extradata: &[u8],
    ) -> Result<Self, Error> {
        let name = CString::new(codec).map_err(|_| Error::NoDecoder(codec.into()))?;
        unsafe {
            let dec = (ff.avcodec_find_decoder_by_name)(name.as_ptr());
            if dec.is_null() {
                return Err(Error::NoDecoder(codec.into()));
            }
            let mut d = Decoder {
                ff,
                ctx: (ff.avcodec_alloc_context3)(dec),
                pkt: (ff.av_packet_alloc)(),
                frame: (ff.av_frame_alloc)(),
                channels: channels as usize,
            };
            if d.ctx.is_null() || d.pkt.is_null() || d.frame.is_null() {
                return Err(Error::Call("allocation", -12));
            }
            if !extradata.is_empty() {
                let mut par = (ff.avcodec_parameters_alloc)();
                if par.is_null() {
                    return Err(Error::Call("avcodec_parameters_alloc", -12));
                }
                let buf = (ff.av_mallocz)(extradata.len() + PADDING) as *mut u8;
                if !buf.is_null() {
                    buf.copy_from_nonoverlapping(extradata.as_ptr(), extradata.len());
                    set_field(par, PAR_TYPE, AVMEDIA_TYPE_AUDIO);
                    set_field(par, PAR_ID, field::<c_int>(dec, CODEC_ID));
                    set_field(par, PAR_EXTRADATA, buf);
                    set_field(par, PAR_EXTRADATA_SIZE, extradata.len() as c_int);
                }
                // par owns buf now; freeing par frees it.
                let r = if buf.is_null() {
                    -12
                } else {
                    (ff.avcodec_parameters_to_context)(d.ctx, par)
                };
                (ff.avcodec_parameters_free)(&mut par);
                if r < 0 {
                    return Err(Error::Call("avcodec_parameters_to_context", r));
                }
            }
            d.opt_int("ar", sample_rate as i64)?;
            let layout = match channels {
                1 => "mono".to_string(),
                2 => "stereo".to_string(),
                n => format!("{n}c"),
            };
            let key = CString::new("ch_layout").unwrap();
            let val = CString::new(layout).unwrap();
            if (ff.av_opt_set)(d.ctx, key.as_ptr(), val.as_ptr(), 0) < 0 {
                d.opt_int("ac", channels as i64)?; // FFmpeg before 5.1
            }
            let r = (ff.avcodec_open2)(d.ctx, dec, std::ptr::null_mut());
            if r < 0 {
                return Err(Error::Call("avcodec_open2", r));
            }
            Ok(d)
        }
    }

    /// FFmpeg's XMA2 decoder for one stream of 1 or 2 channels.
    pub fn xma2(ff: &'a Ffmpeg, sample_rate: u32, channels: u16) -> Result<Self, Error> {
        Self::open(
            ff,
            "xma2",
            sample_rate,
            channels,
            &crate::xma::xma2_extradata(channels),
        )
    }

    fn opt_int(&mut self, key: &str, v: i64) -> Result<(), Error> {
        let k = CString::new(key).unwrap();
        let r = unsafe { (self.ff.av_opt_set_int)(self.ctx, k.as_ptr(), v, 0) };
        if r < 0 {
            return Err(Error::Call("av_opt_set_int", r));
        }
        Ok(())
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Feeds one packet (for XMA2, one or more whole 2048-byte packets) and
    /// appends whatever samples come out.
    pub fn decode(&mut self, packet: &[u8], out: &mut Vec<f32>) -> Result<(), Error> {
        if packet.is_empty() {
            return Ok(()); // an empty packet would mean end of stream
        }
        let len = c_int::try_from(packet.len()).map_err(|_| Error::Call("av_new_packet", -22))?;
        unsafe {
            let r = (self.ff.av_new_packet)(self.pkt, len);
            if r < 0 {
                return Err(Error::Call("av_new_packet", r));
            }
            field::<*mut u8>(self.pkt, PKT_DATA)
                .copy_from_nonoverlapping(packet.as_ptr(), packet.len());
            let r = (self.ff.avcodec_send_packet)(self.ctx, self.pkt);
            (self.ff.av_packet_unref)(self.pkt);
            if r < 0 {
                return Err(Error::Call("avcodec_send_packet", r));
            }
        }
        self.drain(out)
    }

    /// Ends the stream and appends the samples the decoder still holds.
    /// The decoder takes no more packets afterwards.
    pub fn finish(&mut self, out: &mut Vec<f32>) -> Result<(), Error> {
        let r = unsafe { (self.ff.avcodec_send_packet)(self.ctx, std::ptr::null_mut()) };
        if r < 0 && r != EOF {
            return Err(Error::Call("avcodec_send_packet", r));
        }
        self.drain(out)
    }

    fn drain(&mut self, out: &mut Vec<f32>) -> Result<(), Error> {
        loop {
            let r = unsafe { (self.ff.avcodec_receive_frame)(self.ctx, self.frame) };
            if r == EAGAIN || r == EOF {
                return Ok(());
            }
            if r < 0 {
                return Err(Error::Call("avcodec_receive_frame", r));
            }
            let res = unsafe { self.append_frame(out) };
            unsafe { (self.ff.av_frame_unref)(self.frame) };
            res?;
        }
    }

    unsafe fn append_frame(&self, out: &mut Vec<f32>) -> Result<(), Error> {
        let n = field::<c_int>(self.frame, FRAME_NB_SAMPLES).max(0) as usize;
        let fmt = field::<c_int>(self.frame, FRAME_FORMAT);
        let planes = field::<*const *const u8>(self.frame, FRAME_EXTENDED_DATA);
        let ch = self.channels;
        let plane = |i: usize| *planes.add(i);
        out.reserve(n * ch);
        match fmt {
            1 => {
                let p = plane(0) as *const i16; // S16
                out.extend((0..n * ch).map(|i| p.add(i).read_unaligned() as f32 / 32768.0));
            }
            3 => {
                let p = plane(0) as *const f32; // FLT
                out.extend((0..n * ch).map(|i| p.add(i).read_unaligned()));
            }
            6 | 8 => {
                for s in 0..n {
                    for c in 0..ch {
                        out.push(if fmt == 6 {
                            (plane(c) as *const i16).add(s).read_unaligned() as f32 / 32768.0
                        // S16P
                        } else {
                            (plane(c) as *const f32).add(s).read_unaligned() // FLTP
                        });
                    }
                }
            }
            f => return Err(Error::Call("unsupported sample format", f)),
        }
        Ok(())
    }
}

impl Drop for Decoder<'_> {
    fn drop(&mut self) {
        unsafe {
            (self.ff.av_frame_free)(&mut self.frame);
            (self.ff.av_packet_free)(&mut self.pkt);
            (self.ff.avcodec_free_context)(&mut self.ctx);
        }
    }
}
