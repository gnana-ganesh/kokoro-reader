//! Optional candle backend (feature `candle`): runs Kokoro-82M natively on Apple Metal
//! (KOKORO_DEVICE=metal) or on candle's CPU backend (KOKORO_DEVICE=candle-cpu), without ONNX.
//!
//! The model itself is `omp-voice-kokoro` (MIT, Kyle Kelley / Can Boluk; candle port of
//! hexgrad/kokoro, Apache-2.0). Weights are hexgrad's kokoro-v1_0.pth flattened to
//! safetensors by `bench/convert_safetensors.py`.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use omp_voice_kokoro::{KModel, ModelConfig, SynthesisMode};

pub struct CandleModel {
    model: KModel,
    device: Device,
    mode: SynthesisMode,
}

impl CandleModel {
    pub fn load(weights: &str, device: &str) -> Result<(CandleModel, String), String> {
        let dev = match device {
            "metal" => Device::new_metal(0).map_err(|e| format!("Metal device: {e}"))?,
            _ => Device::Cpu,
        };
        let cfg_path = std::path::Path::new(weights).with_file_name("config.json");
        let cfg_txt = std::fs::read_to_string(&cfg_path).map_err(|e| format!("{}: {e}", cfg_path.display()))?;
        let cfg: ModelConfig = serde_json::from_str(&cfg_txt).map_err(|e| format!("config.json: {e}"))?;
        // SAFETY: the safetensors file is mmapped read-only and not modified while loaded.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &dev) }
            .map_err(|e| format!("{weights}: {e}"))?;
        let model = KModel::load(&cfg, vb).map_err(|e| format!("candle load: {e}"))?;
        let mode = if std::env::var("KOKORO_CANDLE_DETERMINISTIC").map(|v| v == "1").unwrap_or(false) {
            SynthesisMode::Deterministic
        } else {
            SynthesisMode::Stochastic
        };
        let name = if dev.is_metal() { "candle-metal" } else { "candle-cpu" };
        Ok((CandleModel { model, device: dev, mode }, name.into()))
    }

    /// `tokens` without the BOS/EOS pads (KModel adds them).
    pub fn infer(&self, tokens: &[i64], style: &[f32], speed: f32) -> Result<Vec<f32>, String> {
        let s = Tensor::from_slice(style, (1, style.len()), &self.device)
            .map_err(|e| e.to_string())?;
        let out = self
            .model
            .forward_with_mode(tokens, &s, speed, &self.device, self.mode)
            .map_err(|e| e.to_string())?;
        out.to_dtype(DType::F32).and_then(|t| t.to_vec1::<f32>()).map_err(|e| e.to_string())
    }
}
