# Credits

This crate is MIT-licensed (same as kokoro-reader). It builds on:

- **hexgrad/Kokoro-82M** — model weights and voices (Apache-2.0). https://huggingface.co/hexgrad/Kokoro-82M
  `src/vocab.rs` is generated from its `config.json`.
- **hexgrad/misaki** — the G2P the model was trained with (Apache-2.0). Lexicon entries in
  `src/normalize.rs` use misaki's phoneme alphabet.
- **thewh1teagle/kokoro-onnx** — ONNX exports and `voices-v1.0.bin` (MIT); the input layout and
  voice-pack indexing follow it.
- **lucasjinreal/Kokoros** — Rust Kokoro inference (Apache-2.0); the npz voice loading and ort
  session approach are adapted from it.
- **Awakened-Labs/loqui `loqui-g2p`** — pure-Rust misaki port, default G2P (MIT).
- **misaki-rs** (MIT) and **espeak-rs / espeak-ng** (espeak-ng is GPL-3.0) — optional G2P backends,
  only compiled with `--features g2p-misaki-rs` / `g2p-espeak`. A binary built with `g2p-espeak`
  links GPL code; the default build does not.
- **pykeio/ort** — ONNX Runtime bindings (MIT/Apache-2.0).
- **omp-voice-kokoro** 0.1.0 (Kyle Kelley, Can Boluk; https://github.com/can1357/omp) — candle/Metal
  Kokoro port (MIT), vendored in `vendor/omp-voice-kokoro` with local changes listed in its
  `VENDORED.md`. Only compiled with `--features candle`.
- **huggingface/candle** — tensor library with the Metal backend (MIT/Apache-2.0).
