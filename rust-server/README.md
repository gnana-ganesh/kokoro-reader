# kokoro-reader-server (Rust prototype)

A Rust drop-in replacement for [`../server.py`](../server.py): same HTTP API, same port, same CORS
rules, so the Chrome extension works unchanged. Built on **axum** + **ONNX Runtime** (`ort`
2.0.0-rc.15) running the Kokoro-82M v1.0 ONNX export, with a pure-Rust misaki port
(**loqui-g2p**) for phonemes. Status: **prototype** – it works and passes the API parity suite, but
on Apple Silicon the Python server on MPS is still faster (see benchmarks).

## Quick start

```bash
cd rust-server
./download_models.sh            # fp32 model + voices -> ~/.cache/kokoro-reader (add: fp16|int8|all)
cargo build --release            # Linux / CPU
cargo build --release --features coreml   # macOS (CoreML EP available, see gaps)
KOKORO_PORT=8881 ./target/release/kokoro-reader-server   # default port is 8880, like server.py
```

`./start.sh` does build + download + run. Stop the Python server first if you want the Rust one on
8880. Extra CLI tools: `kokoro-reader-server phonemize "text"` and
`kokoro-reader-server synth --voice af_heart --out out.wav "text"`.

## API (identical to server.py)

| | |
|---|---|
| `GET /health` | `{"status","device","voices","sample_rate"}` |
| `GET /voices` | the 10 voices (af_heart default, af_bella, af_nicole, af_sarah, am_michael, am_adam, bf_emma, bf_isabella, bm_george, bm_lewis) |
| `POST /tts` | `{text, voice, speed}` → 24 kHz mono 16-bit WAV, `X-Audio-Duration`, `X-Synth-Seconds` |
| CORS | only `chrome-extension://<32 a-p letters>`; no `Origin` allowed; any other Origin → 403 `{"detail":"origin not allowed"}` |

Validation errors are pydantic-shaped 422s; `bench/parity.py OLD_URL NEW_URL` runs 24 cases
(status, headers incl. Vary/preflight, WAV format, error bodies) against both servers:
**24/24 identical** on Linux; 22/24 on the Mac where the only diffs are `/health` `"device"`
(`mps` vs `cpu`).

## Configuration (env)

| var | default | |
|---|---|---|
| `KOKORO_PORT` | 8880 | always binds 127.0.0.1 |
| `KOKORO_MODEL_DIR` | `~/.cache/kokoro-reader` | |
| `KOKORO_MODEL_VARIANT` | fp32 | fp16, int8 |
| `KOKORO_DEVICE` | cpu | `coreml` (needs `--features coreml`) |
| `KOKORO_COREML_UNITS` / `KOKORO_COREML_FORMAT` | all / mlprogram | gpu, ane, cpu / nn |
| `KOKORO_THREADS` | 0 (ORT default) | intra-op threads |
| `KOKORO_G2P` | loqui | `misaki-rs`, `espeak` (need features `g2p-misaki-rs`, `g2p-espeak`) |
| `KOKORO_TECH_LEXICON` | 1 | ISO-date + tech-term pronunciation fixes (`src/normalize.rs`) |
| `KOKORO_LEXICON` | – | extra `word<TAB>misaki-phonemes` TSV |
| `KOKORO_CHUNK_PHONEMES` | 510 | max phonemes per model call |
| `KOKORO_CPU_ARENA` / `KOKORO_MEM_PATTERN` | 1 / 0 | ORT memory knobs (arena off: -40% RSS, ~2x slower) |

## How it works

1. Clean text like server.py (NBSP, whitespace, 6000-char cap).
2. Split into sentences, keeping `?`/`!` and not splitting on `Dr.`, `e.g.`, `3.50`, `v1.2.3`, URLs.
3. Optional normalisation: `2024-03-15` → "March 15, 2024"; `nginx`, `PostgreSQL`, `SQLite`,
   `MySQL`, `PyTorch`, `NumPy`, `kubectl`, `Redis`, `Kubernetes` get fixed phonemes via misaki markup.
4. G2P per sentence (loqui-g2p, American or British by voice prefix), pack sentences into ≤510-phoneme chunks.
5. Tokenise with the **config.json vocab** (`src/vocab.rs`), style vector = `voice[len(phonemes)-1]`,
   run ONNX once per chunk, concatenate, write PCM16 WAV. The model is loaded once at start-up and warmed up.

## Benchmarks

53-word paragraph (`bench/para53.txt`, 23.7 s of audio, af_heart), median of 5 (box) / 3 (Mac)
warm runs, `bench/bench.py`.

**Linux box (8 vCPU x86, CPU only)**

| server | latency | × real-time | RSS (peak) |
|---|---|---|---|
| Python server.py (torch CPU) | 3.66 s | 6.5× | 1456 MB (1885) |
| **Rust fp32** (default knobs) | **1.99–2.25 s** | **10.5–11.9×** | **1009 MB** (1009) |
| Rust fp16 | 1.78 s | 13.5× | 877 MB |
| Rust int8 | 2.63 s | 8.9× | 1002 MB |
| Rust fp32, arena off | 3.78 s | 6.3× | 639 MB (1047) |

Long text (636 words, 280 s audio): Python 42.9 s / 2.0 GB peak → Rust 23.3 s / 1.28 GB.
Start-up to "ready": Python 2.2 s, Rust 1.8 s. G2P is ~2 ms per paragraph.

**MacBook Air M2, 8 GB, macOS 26** (Python production server running on MPS at the same time;
the machine was swapping ~4.5 GB, so treat Mac numbers as ±20 %)

| server | latency | × real-time | phys_footprint | start-up |
|---|---|---|---|---|
| Python server.py, **MPS** | **2.35 s** | 10.1× | 1870 MB | – |
| Python server.py, CPU (earlier runs) | 5–7.5 s | ~4× | ~1.2 GB RSS | – |
| Rust CPU fp32 | 5.92 s | 4.0× | 2231 MB | 5.3 s |
| Rust CPU fp16 | 4.97 s | 4.8× | 2142 MB | 3.2 s |
| Rust CPU int8 | 3.69 s | 6.4× | 2440 MB | 3.6 s |
| Rust CoreML (MLProgram, all/gpu/ane) | fails at inference | – | – | 7–11 s |
| Rust CoreML (NeuralNetwork) | 6.76 s | 3.5× | 3377 MB | 5.2 s |

Release binary: 42.5 MB (macOS arm64, with CoreML), 48 MB (Linux, default features).
`bench/mac_bench.sh` and `bench/mac_tune.sh` reproduce the Mac runs; `bench/results/` holds raw output.

**Take-away:** on x86/Linux CPU the Rust server is ~1.8× faster with ~30–45 % less memory than
the Python server. On the M2 it beats Python-on-CPU (int8 ≈ 1.5–2× faster) but **not** Python on
MPS, because ONNX Runtime's CoreML EP can't run Kokoro's dynamic-shape graph as an MLProgram.

## GPU backends (experimental)

| `KOKORO_DEVICE` | build feature | what runs | weights |
|---|---|---|---|
| `metal` | `candle` | native candle port of Kokoro on the Apple GPU (no ONNX); BiLSTMs on the CPU | `kokoro-v1_0.safetensors` + `config.json` (`bench/convert_safetensors.py`) |
| `candle-cpu` | `candle` | same, candle CPU backend (slow; for correctness checks) | same |
| `webgpu` | `webgpu` | ONNX Runtime WebGPU EP (Dawn, Metal on macOS) | ONNX |
| `coreml` | `coreml` | ONNX Runtime CoreML EP; `KOKORO_COREML_STATIC=1` offloads only static-shape nodes | ONNX |

```sh
cargo build --release --features coreml,webgpu,candle
~/kokoro-reader/.venv/bin/python bench/convert_safetensors.py ~/.cache/kokoro-reader   # once, for metal
KOKORO_DEVICE=metal ./target/release/kokoro-reader-server
```

`KOKORO_CANDLE_DETERMINISTIC=1` zeroes the decoder's random phase/noise (reproducible output).
`KOKORO_LSTM=candle` uses the original per-timestep candle LSTM. Mac numbers: `bench/results/mac-m2-gpu/`.

## Pronunciation (10 tricky lines, `bench/tricky.txt`, vs Python misaki 0.9.4)

| G2P | identical lines | mean similarity | ASR WER vs Python audio* |
|---|---|---|---|
| **loqui-g2p** (default) | 6/10 | 0.975 | 0.088 |
| misaki-rs | 0/10 | 0.751 | 0.175 |
| espeak-ng (what Kokoros uses) | 0/10 | 0.754 | 0.149 |

\*faster-whisper small.en transcript of each server's audio compared with the transcript of the
Python server's audio (`bench/asr.py`). Remaining loqui differences: `St.` (spelled out instead
of "saint"), `I'd've` (drops 've, reads "read" as present tense), missing space in
`std::vector<int>`, O'Neil stress, and neural-OOV guesses for `PostgreSQL`/`Kubernetes`
(the last two are fixed by the built-in tech lexicon). misaki-rs emits stray periods, ZWJ
characters, `£`→❓ and reads years as cardinals; espeak uses non-misaki vowels, reads `:` as
"colon", `1990s` as "nineteen hundred ninety z" and `e.g.` as "for example".

## Known gaps

- **CoreML:** MLProgram fails (`Invalid tensor rank 0 inferred from ios18.squeeze`, unbounded
  input dims); NeuralNetwork format runs but is slower than CPU. Next step: a static-shape /
  bucketed ONNX export, or a Metal backend (candle / MLX), to match Python-on-MPS.
- Memory on macOS (phys_footprint) is not lower than Python yet; the ORT arena dominates.
- No streaming: `/tts` returns one WAV per request, like server.py.
- English voices only (a/b); other Kokoro languages are not wired up.
- macOS link needs `build.rs` (links `libclang_rt.osx.a` for `___isPlatformVersionAtLeast`) and
  `MACOSX_DEPLOYMENT_TARGET=13.4` (`.cargo/config.toml`); both are already in this crate.

## Credits

See [NOTICE.md](NOTICE.md): Kokoro-82M & misaki (hexgrad, Apache-2.0), kokoro-onnx (MIT),
Kokoros (Apache-2.0), loqui-g2p (MIT), ort. The default build contains no GPL code; `--features
g2p-espeak` links espeak-ng (GPL-3.0).
