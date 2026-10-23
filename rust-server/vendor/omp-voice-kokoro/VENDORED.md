# Vendored: omp-voice-kokoro 0.1.0

Source: crates.io `omp-voice-kokoro` 0.1.0 (https://github.com/can1357/omp), MIT License (see LICENSE),
by Kyle Kelley and Can Boluk; itself a candle port of hexgrad/kokoro (Apache-2.0).

Local changes for kokoro-reader (all marked "kokoro-reader change" in the source):
- `src/bilstm.rs`: the recurrence always runs on the CPU (one input-projection matmul, then an f32 matvec loop,
  forward/backward on two threads) instead of per-timestep candle ops on the model device.
  `KOKORO_LSTM=candle` restores the original path.
- `src/model.rs`: duration alignment uses `index_select` on a frame->token index instead of
  building a dense one-hot matrix row by row and multiplying by it.
- Cargo.toml: standalone (no workspace); lints table dropped.
