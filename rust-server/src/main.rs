//! Kokoro Reader TTS server, Rust edition. Drop-in replacement for ../server.py:
//!
//!   GET  /health  -> {"status":"ok","device":"cpu","voices":10,"sample_rate":24000}
//!   GET  /voices  -> {"default":"af_heart","voices":[{"id":..,"label":..}]}
//!   POST /tts     {"text": "...", "voice": "af_heart", "speed": 1.0}  -> audio/wav (24 kHz mono PCM16)
//!
//! Listens on 127.0.0.1 only. Browser requests are accepted only from chrome-extension:// origins.

#[cfg(feature = "candle")]
mod candle_backend;
mod engine;
mod g2p;
mod normalize;
mod text;
mod vocab;

use std::sync::Arc;
use std::time::Instant;

use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;
use serde_json::{json, Value};

use engine::{Engine, EngineOptions, SAMPLE_RATE};

const DEFAULT_VOICES: &[(&str, &str)] = &[
    ("af_heart", "Heart (US female)"),
    ("af_bella", "Bella (US female)"),
    ("af_nicole", "Nicole (US female, soft)"),
    ("af_sarah", "Sarah (US female)"),
    ("am_michael", "Michael (US male)"),
    ("am_adam", "Adam (US male)"),
    ("bf_emma", "Emma (UK female)"),
    ("bf_isabella", "Isabella (UK female)"),
    ("bm_george", "George (UK male)"),
    ("bm_lewis", "Lewis (UK male)"),
];
const EXPOSE_HEADERS: &str = "X-Audio-Duration, X-Synth-Seconds";
const ALLOW_METHODS: &str = "GET, POST, OPTIONS";
const PREFLIGHT_VARY: &str =
    "Origin, Access-Control-Request-Method, Access-Control-Request-Headers, Access-Control-Request-Private-Network";

struct AppState {
    engine: Engine,
    voices: Vec<String>,
    default_voice: String,
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| default.to_string())
}

fn home() -> String {
    std::env::var("HOME").unwrap_or_else(|_| ".".into())
}

fn options() -> EngineOptions {
    let dir = env_or("KOKORO_MODEL_DIR", &format!("{}/.cache/kokoro-reader", home()));
    let variant = env_or("KOKORO_MODEL_VARIANT", "fp32");
    let file = match variant.as_str() {
        "int8" => "kokoro-v1.0.int8.onnx",
        "fp16" => "kokoro-v1.0.fp16.onnx",
        _ => "kokoro-v1.0.onnx",
    };
    let default_g2p = g2p::available().first().copied().unwrap_or("none");
    let device = env_or("KOKORO_DEVICE", "cpu").to_lowercase();
    // candle backends (metal / candle-cpu) read hexgrad's weights as safetensors, not ONNX.
    let file = if device == "metal" || device == "candle-cpu" { "kokoro-v1_0.safetensors" } else { file };
    EngineOptions {
        model_path: env_or("KOKORO_MODEL", &format!("{dir}/{file}")),
        voices_path: env_or("KOKORO_VOICES_FILE", &format!("{dir}/voices-v1.0.bin")),
        device,
        threads: env_or("KOKORO_THREADS", "0").parse().unwrap_or(0),
        g2p: env_or("KOKORO_G2P", default_g2p),
        pack_phonemes: env_or("KOKORO_CHUNK_PHONEMES", "510").parse().unwrap_or(510),
        coreml_cache: Some(format!("{dir}/coreml-cache")),
    }
}

fn label(v: &str) -> String {
    DEFAULT_VOICES.iter().find(|(id, _)| *id == v).map(|(_, l)| l.to_string()).unwrap_or_else(|| v.to_string())
}

fn detail(status: StatusCode, msg: impl Into<Value>) -> Response {
    (status, Json(json!({ "detail": msg.into() }))).into_response()
}

// ------------------------------------------------------------------ CORS / origin guard
/// `^chrome-extension://[a-z]{32}$` without pulling in a regex engine.
fn origin_allowed(origin: &str) -> bool {
    match origin.strip_prefix("chrome-extension://") {
        Some(id) => id.len() == 32 && id.bytes().all(|b| b.is_ascii_lowercase()),
        None => false,
    }
}

/// Mirrors server.py: origin_guard (outermost) + Starlette CORSMiddleware(allow_origin_regex,
/// allow_methods=[GET,POST,OPTIONS], allow_headers=["*"], expose_headers=[...]).
async fn cors(req: Request, next: Next) -> Response {
    let origin = req.headers().get(header::ORIGIN).and_then(|o| o.to_str().ok()).map(str::to_string);
    let Some(origin) = origin else {
        // Starlette's CORSMiddleware adds `Vary: Origin` to every non-preflight response.
        let mut resp = next.run(req).await;
        resp.headers_mut().append(header::VARY, HeaderValue::from_static("Origin"));
        return resp;
    };
    if !origin_allowed(&origin) {
        return detail(StatusCode::FORBIDDEN, "origin not allowed");
    }
    let origin_hv = HeaderValue::from_str(&origin).unwrap();
    if req.method() == Method::OPTIONS && req.headers().contains_key(header::ACCESS_CONTROL_REQUEST_METHOD) {
        let requested = req.headers().get(header::ACCESS_CONTROL_REQUEST_METHOD).and_then(|v| v.to_str().ok()).unwrap_or("");
        let ok = ["GET", "POST", "OPTIONS"].contains(&requested);
        let mut resp = if ok {
            (StatusCode::OK, [(header::CONTENT_TYPE, "text/plain; charset=utf-8")], "OK").into_response()
        } else {
            (StatusCode::BAD_REQUEST, [(header::CONTENT_TYPE, "text/plain; charset=utf-8")], "Disallowed CORS method").into_response()
        };
        let h = resp.headers_mut();
        h.insert(header::VARY, HeaderValue::from_static(PREFLIGHT_VARY));
        h.insert(header::ACCESS_CONTROL_ALLOW_METHODS, HeaderValue::from_static(ALLOW_METHODS));
        h.insert(header::ACCESS_CONTROL_MAX_AGE, HeaderValue::from_static("600"));
        if let Some(rh) = req.headers().get(header::ACCESS_CONTROL_REQUEST_HEADERS) {
            h.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, rh.clone());
        }
        h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, origin_hv);
        return resp;
    }
    let mut resp = next.run(req).await;
    let h = resp.headers_mut();
    h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, origin_hv);
    h.insert(header::ACCESS_CONTROL_EXPOSE_HEADERS, HeaderValue::from_static(EXPOSE_HEADERS));
    h.append(header::VARY, HeaderValue::from_static("Origin"));
    resp
}

// ------------------------------------------------------------------ handlers
#[derive(Serialize)]
struct Health<'a> {
    status: &'a str,
    device: &'a str,
    voices: usize,
    sample_rate: u32,
}

async fn health(State(st): State<Arc<AppState>>) -> Response {
    Json(Health { status: "ok", device: &st.engine.device, voices: st.voices.len(), sample_rate: SAMPLE_RATE }).into_response()
}

#[derive(Serialize)]
struct VoiceItem {
    id: String,
    label: String,
}
#[derive(Serialize)]
struct Voices<'a> {
    default: &'a str,
    voices: Vec<VoiceItem>,
}

async fn voices(State(st): State<Arc<AppState>>) -> Response {
    Json(Voices {
        default: &st.default_voice,
        voices: st.voices.iter().map(|v| VoiceItem { id: v.clone(), label: label(v) }).collect(),
    })
    .into_response()
}

fn err422(kind: &str, loc: Value, msg: &str, input: Value) -> Response {
    detail(StatusCode::UNPROCESSABLE_ENTITY, json!([{ "type": kind, "loc": loc, "msg": msg, "input": input }]))
}

/// Validate like pydantic: text: str (required), voice: str = default, speed: float in [0.5, 2.0].
fn parse_tts(body: &[u8], default_voice: &str) -> Result<(String, String, f32), Response> {
    let v: Value = serde_json::from_slice(body).map_err(|e| {
        detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            json!([{ "type": "json_invalid", "loc": ["body", e.column()], "msg": "JSON decode error", "input": {}, "ctx": {"error": e.to_string()} }]),
        )
    })?;
    let Some(obj) = v.as_object() else {
        return Err(err422("model_attributes_type", json!(["body"]), "Input should be a valid dictionary or object to extract fields from", v.clone()));
    };
    let text = match obj.get("text") {
        None => return Err(err422("missing", json!(["body", "text"]), "Field required", v.clone())),
        Some(Value::String(s)) => s.clone(),
        Some(other) => return Err(err422("string_type", json!(["body", "text"]), "Input should be a valid string", other.clone())),
    };
    let voice = match obj.get("voice") {
        None => default_voice.to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => return Err(err422("string_type", json!(["body", "voice"]), "Input should be a valid string", other.clone())),
    };
    let speed = match obj.get("speed") {
        None => 1.0,
        Some(Value::Number(n)) => n.as_f64().unwrap_or(1.0),
        Some(Value::String(s)) if s.trim().parse::<f64>().is_ok() => s.trim().parse::<f64>().unwrap(),
        Some(other) => return Err(err422("float_parsing", json!(["body", "speed"]), "Input should be a valid number", other.clone())),
    };
    if speed < 0.5 {
        return Err(detail(StatusCode::UNPROCESSABLE_ENTITY, json!([{ "type": "greater_than_equal", "loc": ["body", "speed"], "msg": "Input should be greater than or equal to 0.5", "input": obj["speed"], "ctx": {"ge": 0.5} }])));
    }
    if speed > 2.0 {
        return Err(detail(StatusCode::UNPROCESSABLE_ENTITY, json!([{ "type": "less_than_equal", "loc": ["body", "speed"], "msg": "Input should be less than or equal to 2", "input": obj["speed"], "ctx": {"le": 2.0} }])));
    }
    Ok((text, voice, speed as f32))
}

fn py_list(items: &[String]) -> String {
    format!("[{}]", items.iter().map(|s| format!("'{s}'")).collect::<Vec<_>>().join(", "))
}

async fn tts(State(st): State<Arc<AppState>>, body: Bytes) -> Response {
    let (raw, voice, speed) = match parse_tts(&body, &st.default_voice) {
        Ok(x) => x,
        Err(r) => return r,
    };
    let text_clean = text::clean(&raw);
    if text_clean.is_empty() {
        return detail(StatusCode::BAD_REQUEST, "empty text");
    }
    if !st.voices.contains(&voice) {
        return detail(StatusCode::BAD_REQUEST, format!("unknown voice '{voice}'; available: {}", py_list(&st.voices)));
    }
    let t = Instant::now();
    let st2 = st.clone();
    let (text2, voice2) = (text_clean.clone(), voice.clone());
    let res = tokio::task::spawn_blocking(move || st2.engine.synthesize(&text2, &voice2, speed)).await;
    let synth = match res {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => return detail(StatusCode::INTERNAL_SERVER_ERROR, format!("synthesis failed: {e}")),
        Err(e) => return detail(StatusCode::INTERNAL_SERVER_ERROR, format!("synthesis failed: {e}")),
    };
    let took = t.elapsed().as_secs_f64();
    let audio = if synth.audio.is_empty() { vec![0.0; (SAMPLE_RATE as f32 * 0.2) as usize] } else { synth.audio };
    let dur = audio.len() as f64 / SAMPLE_RATE as f64;
    eprintln!(
        "[kokoro-rs] {} words, {voice} x{speed}: {dur:.1}s audio in {took:.2}s (g2p {:.3}s, infer {:.2}s, {} chunk(s))",
        text_clean.split_whitespace().count(),
        synth.g2p_secs,
        synth.infer_secs,
        synth.chunks
    );
    let wav = engine::wav_pcm16(&audio, SAMPLE_RATE);
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
    headers.insert("X-Audio-Duration", HeaderValue::from_str(&format!("{dur:.2}")).unwrap());
    headers.insert("X-Synth-Seconds", HeaderValue::from_str(&format!("{took:.2}")).unwrap());
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    (StatusCode::OK, headers, Body::from(wav)).into_response()
}

async fn not_found() -> Response {
    detail(StatusCode::NOT_FOUND, "Not Found")
}
async fn method_not_allowed() -> Response {
    detail(StatusCode::METHOD_NOT_ALLOWED, "Method Not Allowed")
}

// ------------------------------------------------------------------ main
fn usage() -> ! {
    eprintln!(
        "usage:\n  kokoro-reader-server                     run the HTTP server (127.0.0.1:$KOKORO_PORT, default 8880)\n  kokoro-reader-server phonemize [--g2p NAME] [--lang a|b] TEXT\n  kokoro-reader-server synth [--voice V] [--speed S] [--out FILE] TEXT\n  kokoro-reader-server profile [--runs N] [--voice V] [--out FILE] TEXT   (KOKORO_PROFILE=1: per-stage)\n\nenv: KOKORO_PORT KOKORO_DEVICE=cpu|coreml KOKORO_THREADS KOKORO_MODEL_DIR KOKORO_MODEL_VARIANT=fp32|fp16|int8\n     KOKORO_MODEL KOKORO_VOICES_FILE KOKORO_G2P={:?} KOKORO_DEFAULT_VOICE KOKORO_VOICES=all KOKORO_CHUNK_PHONEMES",
        g2p::available()
    );
    std::process::exit(2)
}

fn flag(args: &mut Vec<String>, name: &str) -> Option<String> {
    let i = args.iter().position(|a| a == name)?;
    if i + 1 >= args.len() {
        usage();
    }
    let v = args.remove(i + 1);
    args.remove(i);
    Some(v)
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        usage();
    }
    match args.first().map(String::as_str) {
        Some("phonemize") => {
            args.remove(0);
            let name = flag(&mut args, "--g2p").unwrap_or_else(|| env_or("KOKORO_G2P", g2p::available()[0]));
            let lang = if flag(&mut args, "--lang").as_deref() == Some("b") { g2p::Lang::British } else { g2p::Lang::American };
            let p = g2p::build(&name).unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(1) });
            let input = if args.is_empty() { std::io::read_to_string(std::io::stdin()).unwrap() } else { args.join(" ") };
            for line in input.lines().filter(|l| !l.trim().is_empty()) {
                let line = if normalize::enabled() && p.supports_markup() { normalize::apply(line) } else { line.to_string() };
                match p.phonemize(&line, lang) {
                    Ok(ps) => println!("{ps}"),
                    Err(e) => println!("ERROR: {e}"),
                }
            }
            return;
        }
        Some("profile") => {
            // profile [--runs N] [--voice V] [--out FILE] TEXT: 1 warm-up + N runs; with
            // KOKORO_PROFILE=1 on a candle device, also per-stage medians (device-synced).
            args.remove(0);
            let voice = flag(&mut args, "--voice").unwrap_or_else(|| "af_heart".into());
            let runs: usize = flag(&mut args, "--runs").map(|s| s.parse().unwrap()).unwrap_or(3);
            let out = flag(&mut args, "--out");
            let t0 = Instant::now();
            let eng = Engine::load(&options()).unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(1) });
            let load_s = t0.elapsed().as_secs_f64();
            let txt = text::clean(&args.join(" "));
            let mut totals = Vec::new();
            let mut stages: Vec<(String, Vec<f64>)> = Vec::new();
            let mut first_s = 0.0;
            let mut audio_s = 0.0;
            for i in 0..=runs {
                let t = Instant::now();
                let s = eng.synthesize(&txt, &voice, 1.0).unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(1) });
                let el = t.elapsed().as_secs_f64();
                audio_s = s.audio.len() as f64 / SAMPLE_RATE as f64;
                let laps = engine::take_profile();
                if i == 0 {
                    first_s = el;
                    if let Some(o) = &out { std::fs::write(o, engine::wav_pcm16(&s.audio, SAMPLE_RATE)).unwrap(); }
                    continue;
                }
                totals.push(el);
                for (l, v) in laps {
                    match stages.iter_mut().find(|e| e.0 == l) {
                        Some(e) => e.1.push(v),
                        None => stages.push((l.to_string(), vec![v])),
                    }
                }
            }
            let med = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
            let total = med(&mut totals);
            let mut st = serde_json::Map::new();
            let mut ordered = stages;
            ordered.sort_by(|a, b| a.0.cmp(&b.0));
            for (l, mut v) in ordered {
                st.insert(l, serde_json::json!((med(&mut v) * 1000.0).round() / 1000.0));
            }
            println!("{}", serde_json::json!({"device": eng.device, "load_s": (load_s*100.0).round()/100.0,
                "first_s": (first_s*100.0).round()/100.0, "median_s": (total*1000.0).round()/1000.0,
                "audio_s": (audio_s*100.0).round()/100.0, "stages_s": st}));
            return;
        }
        Some("synth") => {
            args.remove(0);
            let voice = flag(&mut args, "--voice").unwrap_or_else(|| "af_heart".into());
            let speed: f32 = flag(&mut args, "--speed").map(|s| s.parse().unwrap()).unwrap_or(1.0);
            let out = flag(&mut args, "--out").unwrap_or_else(|| "out.wav".into());
            let t0 = Instant::now();
            let eng = Engine::load(&options()).unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(1) });
            eprintln!("loaded in {:.2}s", t0.elapsed().as_secs_f64());
            let txt = text::clean(&args.join(" "));
            for ps in eng.plan(&txt, g2p::Lang::for_voice(&voice)).unwrap() {
                eprintln!("chunk ({} phonemes): {ps}", eng.tokenize(&ps).len());
            }
            let t1 = Instant::now();
            let s = eng.synthesize(&txt, &voice, speed).unwrap();
            eprintln!("{:.2}s audio in {:.2}s", s.audio.len() as f64 / SAMPLE_RATE as f64, t1.elapsed().as_secs_f64());
            std::fs::write(&out, engine::wav_pcm16(&s.audio, SAMPLE_RATE)).unwrap();
            return;
        }
        Some(other) => {
            eprintln!("unknown command {other:?}");
            usage();
        }
        None => {}
    }

    let opts = options();
    for p in [&opts.model_path, &opts.voices_path] {
        if !engine::exists(p) {
            eprintln!("[kokoro-rs] missing {p}. Run rust-server/download_models.sh first (see rust-server/README.md).");
            std::process::exit(1);
        }
    }
    eprintln!("[kokoro-rs] loading {} on {} (g2p: {}) ...", opts.model_path, opts.device, opts.g2p);
    let t0 = Instant::now();
    let engine = Engine::load(&opts).unwrap_or_else(|e| {
        eprintln!("[kokoro-rs] failed to load: {e}");
        std::process::exit(1)
    });
    let mut voices: Vec<String> = if env_or("KOKORO_VOICES", "") == "all" {
        engine.voices.keys().cloned().collect()
    } else {
        DEFAULT_VOICES.iter().map(|(v, _)| v.to_string()).filter(|v| engine.voices.contains_key(v)).collect()
    };
    voices.sort();
    let default_voice = env_or("KOKORO_DEFAULT_VOICE", "af_heart");
    if let Err(e) = engine.synthesize("Ready.", &default_voice, 1.0) {
        eprintln!("[kokoro-rs] warm-up failed: {e}");
    }
    eprintln!(
        "[kokoro-rs] ready in {:.1}s (g2p {}); voices: {}",
        t0.elapsed().as_secs_f64(),
        engine.g2p.name(),
        voices.join(", ")
    );
    let state = Arc::new(AppState { engine, voices, default_voice });

    let app = Router::new()
        .route("/health", get(health))
        .route("/voices", get(voices_handler))
        .route("/tts", post(tts))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .layer(middleware::from_fn(cors))
        .with_state(state);

    let port: u16 = env_or("KOKORO_PORT", "8880").parse().unwrap_or(8880);
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.unwrap_or_else(|e| {
            eprintln!("[kokoro-rs] cannot bind 127.0.0.1:{port}: {e}");
            std::process::exit(1)
        });
        eprintln!("[kokoro-rs] listening on http://127.0.0.1:{port}");
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await
            .unwrap();
    });
}

async fn voices_handler(st: State<Arc<AppState>>) -> Response {
    voices(st).await
}
