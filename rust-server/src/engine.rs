//! Kokoro-82M ONNX inference: model + voice packs loaded once, sentence-level chunking.
//!
//! Inference I/O layout and the voices-v1.0.bin (npz) format follow lucasjinreal/Kokoros
//! (Apache-2.0) and thewh1teagle/kokoro-onnx (MIT); style indexing and the 510-phoneme
//! window follow hexgrad/kokoro (Apache-2.0).

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use ort::session::Session;
use ort::value::Tensor;

use crate::g2p::{Lang, Phonemizer};
use crate::text;
use crate::vocab::VOCAB;

pub const SAMPLE_RATE: u32 = 24_000;
/// Kokoro's context is 512 tokens incl. the two pad tokens.
pub const MAX_PHONEMES: usize = 510;
const STYLE_DIM: usize = 256;

pub struct Voice {
    /// [510][256] style vectors indexed by phoneme count - 1.
    styles: Vec<f32>,
    rows: usize,
}

impl Voice {
    fn style_for(&self, n_phonemes: usize) -> &[f32] {
        let idx = n_phonemes.saturating_sub(1).min(self.rows - 1);
        &self.styles[idx * STYLE_DIM..(idx + 1) * STYLE_DIM]
    }
}

pub struct Engine {
    session: Option<Mutex<Session>>,
    #[cfg(feature = "candle")]
    candle: Option<crate::candle_backend::CandleModel>,
    tokens_name: String,
    style_name: String,
    speed_name: String,
    speed_is_int: bool,
    pub voices: HashMap<String, Voice>,
    pub g2p: Box<dyn Phonemizer>,
    vocab: HashMap<char, i64>,
    pub device: String,
    pub pack_phonemes: usize,
}

pub struct Synth {
    pub audio: Vec<f32>,
    pub chunks: usize,
    pub g2p_secs: f64,
    pub infer_secs: f64,
}

#[derive(Clone, Debug)]
pub struct EngineOptions {
    pub model_path: String,
    pub voices_path: String,
    pub device: String,
    pub threads: usize,
    pub g2p: String,
    pub pack_phonemes: usize,
    #[cfg_attr(not(feature = "coreml"), allow(dead_code))]
    pub coreml_cache: Option<String>,
}

fn build_session(opts: &EngineOptions) -> Result<(Session, String), String> {
    let env = ort::init().with_name("kokoro-reader").build().map_err(|e| e.to_string())?;
    let mut b = Session::builder(&env).map_err(|e| e.to_string())?;
    if opts.threads > 0 {
        b = b.with_intra_threads(opts.threads).map_err(|e| e.to_string())?;
    }
    // Memory knobs. Kokoro has dynamic shapes (token count varies per chunk), so ORT's
    // memory-pattern planner caches a plan per shape and the CPU arena keeps high-water
    // allocations. KOKORO_MEM_PATTERN / KOKORO_CPU_ARENA = 1|0 toggle them (see README table).
    let flag = |k: &str, d: bool| std::env::var(k).map(|v| v == "1").unwrap_or(d);
    let mem_pattern = flag("KOKORO_MEM_PATTERN", false);
    let arena = flag("KOKORO_CPU_ARENA", true);
    b = b.with_memory_pattern(mem_pattern).map_err(|e| e.to_string())?;
    if !arena && opts.device != "coreml" {
        b = b
            .with_execution_providers([ort::ep::CPU::default().with_arena_allocator(false).build()])
            .map_err(|e| e.to_string())?;
    }
    #[allow(unused_mut)]
    let mut device = "cpu".to_string();
    if opts.device == "webgpu" {
        #[cfg(feature = "webgpu")]
        {
            b = b
                .with_execution_providers([ort::ep::WebGPU::default().build().error_on_failure()])
                .map_err(|e| format!("WebGPU execution provider failed: {e}"))?;
            device = "webgpu".into();
        }
        #[cfg(not(feature = "webgpu"))]
        return Err("built without the `webgpu` feature; rebuild with --features webgpu".into());
    }
    if opts.device == "coreml" {
        #[cfg(feature = "coreml")]
        {
            use ort::ep::coreml::{ComputeUnits, ModelFormat};
            let units = match std::env::var("KOKORO_COREML_UNITS").unwrap_or_default().as_str() {
                "gpu" => ComputeUnits::CPUAndGPU,
                "ane" => ComputeUnits::CPUAndNeuralEngine,
                "cpu" => ComputeUnits::CPUOnly,
                _ => ComputeUnits::All,
            };
            let format = match std::env::var("KOKORO_COREML_FORMAT").unwrap_or_default().as_str() {
                "nn" | "neuralnetwork" => ModelFormat::NeuralNetwork,
                _ => ModelFormat::MLProgram,
            };
            let mut ep = ort::ep::CoreML::default()
                .with_model_format(format)
                .with_compute_units(units)
                .with_subgraphs(true);
            // KOKORO_COREML_STATIC=1: only hand CoreML the nodes whose input shapes are static
            // (the rest stays on ORT's CPU EP). KOKORO_COREML_FAST=1: FastPrediction
            // specialization. KOKORO_COREML_LOWPREC=1: fp16 accumulation on the GPU.
            let flag = |k: &str| std::env::var(k).map(|v| v == "1").unwrap_or(false);
            if flag("KOKORO_COREML_STATIC") {
                ep = ep.with_static_input_shapes(true);
            }
            if flag("KOKORO_COREML_FAST") {
                ep = ep.with_specialization_strategy(ort::ep::coreml::SpecializationStrategy::FastPrediction);
            }
            if flag("KOKORO_COREML_LOWPREC") {
                ep = ep.with_low_precision_accumulation_on_gpu(true);
            }
            if let Some(base) = &opts.coreml_cache {
                let tag = if flag("KOKORO_COREML_STATIC") { "-static" } else { "" };
                let dir = format!("{base}/{units:?}-{format:?}{tag}");
                let _ = std::fs::create_dir_all(&dir);
                ep = ep.with_model_cache_dir(dir);
            }
            b = b
                .with_execution_providers([ep.build().error_on_failure()])
                .map_err(|e| format!("CoreML execution provider failed: {e}"))?;
            device = "coreml".into();
        }
        #[cfg(not(feature = "coreml"))]
        return Err("built without the `coreml` feature; rebuild with --features coreml".into());
    }
    let s = b.commit_from_file(&opts.model_path).map_err(|e| format!("loading {}: {e}", opts.model_path))?;
    Ok((s, device))
}

/// Parse voices-v1.0.bin: a numpy .npz (zip of .npy), each voice float32 [510, 1, 256].
pub fn load_voices(path: &str) -> Result<HashMap<String, Voice>, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut zip = zip::ZipArchive::new(f).map_err(|e| format!("{path}: {e}"))?;
    let mut out = HashMap::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(|e| e.to_string())?;
        let name = entry.name().trim_end_matches(".npy").to_string();
        let mut buf = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        if buf.len() < 10 || &buf[..6] != b"\x93NUMPY" {
            return Err(format!("{name}: not an .npy entry"));
        }
        let major = buf[6];
        let (hlen, hstart) = if major == 1 {
            (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10)
        } else {
            (u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize, 12)
        };
        let header = String::from_utf8_lossy(&buf[hstart..hstart + hlen]);
        if !header.contains("'<f4'") || header.contains("'fortran_order': True") {
            return Err(format!("{name}: unsupported npy header {header}"));
        }
        let data = &buf[hstart + hlen..];
        let styles: Vec<f32> = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        if styles.len() % STYLE_DIM != 0 || styles.is_empty() {
            return Err(format!("{name}: unexpected size {}", styles.len()));
        }
        let rows = styles.len() / STYLE_DIM;
        out.insert(name, Voice { styles, rows });
    }
    Ok(out)
}

impl Engine {
    pub fn load(opts: &EngineOptions) -> Result<Engine, String> {
        if opts.device == "metal" || opts.device == "candle-cpu" {
            #[cfg(feature = "candle")]
            {
                let (m, device) = crate::candle_backend::CandleModel::load(&opts.model_path, &opts.device)?;
                return Ok(Engine {
                    session: None,
                    candle: Some(m),
                    tokens_name: String::new(),
                    style_name: String::new(),
                    speed_name: String::new(),
                    speed_is_int: false,
                    voices: load_voices(&opts.voices_path)?,
                    g2p: crate::g2p::build(&opts.g2p)?,
                    vocab: VOCAB.iter().copied().collect(),
                    device,
                    pack_phonemes: opts.pack_phonemes.clamp(16, MAX_PHONEMES),
                });
            }
            #[cfg(not(feature = "candle"))]
            return Err("built without the `candle` feature; rebuild with --features candle".into());
        }
        let (session, device) = build_session(opts)?;
        let names: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
        let pick = |cands: &[&str]| -> Result<String, String> {
            cands
                .iter()
                .find(|c| names.iter().any(|n| n == *c))
                .map(|s| s.to_string())
                .ok_or_else(|| format!("model inputs {names:?} lack one of {cands:?}"))
        };
        let tokens_name = pick(&["tokens", "input_ids"])?;
        let style_name = pick(&["style", "ref_s"])?;
        let speed_name = pick(&["speed"])?;
        let speed_is_int = session
            .inputs()
            .iter()
            .find(|i| i.name() == speed_name)
            .map(|i| format!("{:?}", i.dtype()).contains("Int"))
            .unwrap_or(false);
        let voices = load_voices(&opts.voices_path)?;
        let g2p = crate::g2p::build(&opts.g2p)?;
        Ok(Engine {
            session: Some(Mutex::new(session)),
            #[cfg(feature = "candle")]
            candle: None,
            tokens_name,
            style_name,
            speed_name,
            speed_is_int,
            voices,
            g2p,
            vocab: VOCAB.iter().copied().collect(),
            device,
            pack_phonemes: opts.pack_phonemes.clamp(16, MAX_PHONEMES),
        })
    }

    pub fn tokenize(&self, phonemes: &str) -> Vec<i64> {
        phonemes.chars().filter_map(|c| self.vocab.get(&c).copied()).collect()
    }

    /// Text -> list of phoneme strings, each <= MAX_PHONEMES tokens, packed sentence-wise.
    pub fn plan(&self, text_in: &str, lang: Lang) -> Result<Vec<String>, String> {
        let mut pieces: Vec<String> = Vec::new();
        let markup = crate::normalize::enabled() && self.g2p.supports_markup();
        for sentence in text::split_sentences(text_in) {
            let sentence = if markup { crate::normalize::apply(&sentence) } else { sentence };
            let ps = self.g2p.phonemize(&sentence, lang)?;
            if self.tokenize(&ps).len() <= MAX_PHONEMES {
                pieces.push(ps);
                continue;
            }
            for part in text::split_long(&sentence, 180) {
                let ps = self.g2p.phonemize(&part, lang)?;
                let mut ps: String = ps.chars().filter(|c| self.vocab.contains_key(c)).collect();
                if ps.chars().count() > MAX_PHONEMES {
                    ps = ps.chars().take(MAX_PHONEMES).collect();
                }
                pieces.push(ps);
            }
        }
        // pack consecutive sentences up to pack_phonemes
        let mut chunks: Vec<String> = Vec::new();
        for p in pieces {
            let p = p.trim().to_string();
            if p.is_empty() {
                continue;
            }
            if let Some(last) = chunks.last_mut() {
                if self.tokenize(last).len() + 1 + self.tokenize(&p).len() <= self.pack_phonemes {
                    last.push(' ');
                    last.push_str(&p);
                    continue;
                }
            }
            chunks.push(p);
        }
        Ok(chunks)
    }

    fn infer(&self, tokens: &[i64], style: &[f32], speed: f32) -> Result<Vec<f32>, String> {
        #[cfg(feature = "candle")]
        if let Some(m) = &self.candle {
            return m.infer(tokens, style, speed);
        }
        let mut ids = Vec::with_capacity(tokens.len() + 2);
        ids.push(0i64);
        ids.extend_from_slice(tokens);
        ids.push(0);
        let n = ids.len();
        let t_tokens = Tensor::from_array(([1usize, n], ids)).map_err(|e| e.to_string())?;
        let t_style = Tensor::from_array(([1usize, STYLE_DIM], style.to_vec())).map_err(|e| e.to_string())?;
        let mut sess = self.session.as_ref().ok_or("no ONNX session")?.lock().map_err(|e| e.to_string())?;
        let outputs = if self.speed_is_int {
            let t_speed = Tensor::from_array(([1usize], vec![speed.round() as i32])).map_err(|e| e.to_string())?;
            sess.run(ort::inputs![
                self.tokens_name.as_str() => t_tokens,
                self.style_name.as_str() => t_style,
                self.speed_name.as_str() => t_speed
            ])
        } else {
            let t_speed = Tensor::from_array(([1usize], vec![speed])).map_err(|e| e.to_string())?;
            sess.run(ort::inputs![
                self.tokens_name.as_str() => t_tokens,
                self.style_name.as_str() => t_style,
                self.speed_name.as_str() => t_speed
            ])
        }
        .map_err(|e| e.to_string())?;
        let (_, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
        Ok(data.to_vec())
    }

    pub fn synthesize(&self, text_in: &str, voice: &str, speed: f32) -> Result<Synth, String> {
        let v = self.voices.get(voice).ok_or_else(|| format!("unknown voice {voice}"))?;
        let t0 = Instant::now();
        let chunks = self.plan(text_in, Lang::for_voice(voice))?;
        let g2p_secs = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let mut audio = Vec::new();
        for ps in &chunks {
            let tokens = self.tokenize(ps);
            if tokens.is_empty() {
                continue;
            }
            let style = v.style_for(tokens.len());
            audio.extend(self.infer(&tokens, style, speed)?);
        }
        Ok(Synth { audio, chunks: chunks.len(), g2p_secs, infer_secs: t1.elapsed().as_secs_f64() })
    }
}

/// Per-stage timings recorded since the last call (candle backend with KOKORO_PROFILE=1).
pub fn take_profile() -> Vec<(&'static str, f64)> {
    #[cfg(feature = "candle")]
    return omp_voice_kokoro::prof::take();
    #[cfg(not(feature = "candle"))]
    Vec::new()
}

/// 16-bit PCM mono WAV (what soundfile writes for subtype="PCM_16").
pub fn wav_pcm16(audio: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = (audio.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in audio {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn exists(p: &str) -> bool {
    Path::new(p).exists()
}
