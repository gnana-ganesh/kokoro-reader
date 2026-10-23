//! KModel — the top-level Kokoro TTS model.
//!
//! Ported from kokoro/model.py

use candle_core::{DType, Device, Error, Module, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};

use crate::albert::CustomAlbert;
use crate::config::ModelConfig;
use crate::istftnet::{Decoder, SynthesisMode};
use crate::modules::{ProsodyPredictor, TextEncoder};

/// The top-level Kokoro-82M model.
pub struct KModel {
    bert: CustomAlbert,
    bert_encoder: nn::Linear,
    predictor: ProsodyPredictor,
    text_encoder: TextEncoder,
    decoder: Decoder,
    context_length: usize,
}

impl KModel {
    /// Load model weights from a VarBuilder (typically backed by safetensors).
    pub fn load(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let bert = CustomAlbert::load(&config.plbert, config.n_token, vb.pp("bert"))?;

        let bert_encoder = nn::linear(
            config.plbert.hidden_size,
            config.hidden_dim,
            vb.pp("bert_encoder"),
        )?;

        let predictor = ProsodyPredictor::load(
            config.style_dim,
            config.hidden_dim,
            config.n_layer,
            config.max_dur,
            vb.pp("predictor"),
        )?;

        let text_encoder = TextEncoder::load(
            config.hidden_dim,
            config.text_encoder_kernel_size,
            config.n_layer,
            config.n_token,
            vb.pp("text_encoder"),
        )?;

        let istft = &config.istftnet;
        let decoder = Decoder::load(
            config.hidden_dim,
            config.style_dim,
            config.n_mels,
            &istft.resblock_kernel_sizes,
            &istft.upsample_rates,
            istft.upsample_initial_channel,
            &istft.resblock_dilation_sizes,
            &istft.upsample_kernel_sizes,
            istft.gen_istft_n_fft,
            istft.gen_istft_hop_size,
            vb.pp("decoder"),
        )?;

        Ok(Self {
            bert,
            bert_encoder,
            predictor,
            text_encoder,
            decoder,
            context_length: config.plbert.max_position_embeddings,
        })
    }

    /// Run inference.
    ///
    /// * `input_ids` - phoneme token IDs (without BOS/EOS padding; the model adds [0, ..., 0])
    /// * `ref_s` - reference style tensor [1, 256] (first 128 = acoustic style, last 128 = prosody style)
    /// * `speed` - speed factor (1.0 = normal)
    ///
    /// Returns: audio waveform tensor [samples]
    pub fn forward(
        &self,
        input_ids: &[i64],
        ref_s: &Tensor,
        speed: f32,
        device: &Device,
    ) -> Result<Tensor> {
        self.forward_with_mode(input_ids, ref_s, speed, device, SynthesisMode::Stochastic)
    }

    /// Run inference with explicit synthesis-mode control.
    ///
    /// Deterministic mode removes the decoder's random harmonic phase and noise
    /// sources, making output reproducible on CPU and GPU backends.
    pub fn forward_with_mode(
        &self,
        input_ids: &[i64],
        ref_s: &Tensor,
        speed: f32,
        device: &Device,
        mode: SynthesisMode,
    ) -> Result<Tensor> {
        crate::prof::begin(device);
        // Pad with BOS=0 and EOS=0
        let mut padded = Vec::with_capacity(input_ids.len() + 2);
        padded.push(0i64);
        padded.extend_from_slice(input_ids);
        padded.push(0i64);

        let seq_len = padded.len();
        validate_context_length(seq_len, self.context_length)?;

        let input_ids_t = Tensor::new(&padded[..], device)?.unsqueeze(0)?; // [1, T]
        let input_lengths = Tensor::new(&[seq_len as i64][..], device)?; // [1]

        // Text mask: [1, T], true where position >= length (i.e., padding)
        // Python: mask = torch.gt(mask+1, lengths.unsqueeze(1))
        // For single-sequence inference with no padding, this is all false (0).
        // For single-sequence inference with no padding, mask is all false
        let mask_data: Vec<f32> = (0..seq_len)
            .map(|i| if i + 1 > seq_len { 1.0 } else { 0.0 })
            .collect();
        let text_mask = Tensor::new(&mask_data[..], device)?.unsqueeze(0)?; // [1, T]
        let text_mask = text_mask.gt(0.5)?; // [1, T] u8
        let text_mask_f = text_mask.to_dtype(DType::F32)?;

        // BERT: attention_mask is 1=attend, 0=pad (inverse of text_mask)
        let attention_mask = (1.0 - &text_mask_f)?;

        // ALBERT forward
        let bert_dur = self.bert.forward(
            &input_ids_t.to_dtype(DType::U32)?,
            &attention_mask.to_dtype(DType::F32)?,
        )?; // [1, T, hidden_size]

        crate::prof::lap(device, "1 bert (albert)");
        // bert_encoder: project hidden_size -> hidden_dim
        let d_en = self.bert_encoder.forward(&bert_dur)?; // [1, T, hidden_dim]
        let d_en = d_en.transpose(1, 2)?; // [1, hidden_dim, T]

        // Style split: ref_s[:, 128:] = prosody style, ref_s[:, :128] = acoustic style
        let s = ref_s.narrow(1, 128, 128)?; // [1, 128] prosody style

        // Duration encoder
        let d = self
            .predictor
            .text_encoder
            .forward(&d_en, &s, &input_lengths, &text_mask)?; // [1, T, hidden_dim]

        crate::prof::lap(device, "2 duration encoder (3 bilstm + adaln)");
        // Duration LSTM: needs [1, T, hidden_dim + style_dim]
        let (b, t_dur, _) = d.dims3()?;
        let s_exp = s.unsqueeze(1)?.expand(&[b, t_dur, 128])?;
        let d_cat = Tensor::cat(&[&d, &s_exp], 2)?; // [1, T, hidden_dim + 128]
        let x = self.predictor.lstm.forward(&d_cat)?; // [1, T, hidden_dim]

        // Duration projection
        let duration = self.predictor.duration_proj.forward(&x)?; // [1, T, max_dur]
        let duration = candle_nn::ops::sigmoid(&duration)?;
        let duration = duration.sum(2)?; // [1, T]
        let speed_t = Tensor::new(speed, device)?.to_dtype(duration.dtype())?;
        let duration = duration.broadcast_div(&speed_t)?;
        let pred_dur = duration.round()?.clamp(1.0f32, f32::MAX)?;

        crate::prof::lap(device, "3 duration lstm + proj");
        // Build alignment matrix on GPU via repeat_interleave-style expansion.
        //
        // The alignment matrix is a [T, total_frames] binary matrix where each
        // token row has 1s in the frame columns it occupies. Instead of building
        // this with a CPU loop (which requires GPU→CPU→GPU round-trip), we:
        //   1. Download durations once (small: ~100 i64 values)
        //   2. Build per-row one-hot slices and concatenate on GPU
        //
        // For typical phoneme counts (50-150 tokens), the duration download is
        // ~400 bytes — negligible vs the old approach which also downloaded them.
        let pred_dur_vec: Vec<i64> = pred_dur
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
            .iter()
            .map(|&v| v.max(1.0) as i64)
            .collect();
        let pred_dur_vec = normalize_boundary_token_durations(pred_dur_vec);
        let total_frames: usize = pred_dur_vec.iter().sum::<i64>() as usize;

        // kokoro-reader change: the alignment matrix is one-hot per frame, so `d^T @ aln` is a
        // gather of token columns. One index tensor + index_select replaces the T per-row
        // tensors/cats and the dense [T, frames] matmul (much fewer Metal dispatches).
        let _ = seq_len;
        let mut frame_tok: Vec<u32> = Vec::with_capacity(total_frames);
        for (token_idx, &dur) in pred_dur_vec.iter().enumerate() {
            for _ in 0..dur.max(0) {
                frame_tok.push(token_idx as u32);
            }
        }
        let frame_idx = Tensor::from_vec(frame_tok, total_frames, device)?;

        // en = d.transpose(-1,-2) @ pred_aln_trg
        let d_t = d.transpose(1, 2)?.contiguous()?;
        let en = d_t.index_select(&frame_idx, 2)?; // [1, hidden_dim, total_frames]

        crate::prof::lap(device, "4 alignment (dur d2h + gather)");
        // F0 and N prediction
        let (f0_pred, n_pred) = self.predictor.f0_n_train(&en, &s)?;

        crate::prof::lap(device, "5c F0/N blocks");
        // Text encoder
        let t_en = self.text_encoder.forward(
            &input_ids_t.to_dtype(DType::U32)?,
            &input_lengths,
            &text_mask,
        )?; // [1, hidden_dim, T]

        // asr = t_en @ pred_aln_trg
        let asr = t_en.contiguous()?.index_select(&frame_idx, 2)?; // [1, hidden_dim, total_frames]

        crate::prof::lap(device, "6 text encoder (cnn + bilstm) + gather");
        // Decoder
        let s_acoustic = ref_s.narrow(1, 0, 128)?; // [1, 128]
        let audio = self
            .decoder
            .forward_with_mode(&asr, &f0_pred, &n_pred, &s_acoustic, mode)?;

        let audio = audio.squeeze(0)?.squeeze(0)?;
        crate::prof::lap(device, "8f istft (convT) + trim");
        Ok(audio)
    }
}

fn validate_context_length(seq_len: usize, context_length: usize) -> Result<()> {
    if seq_len > context_length {
        return Err(Error::Msg(format!(
            "Input too long: {seq_len} > {context_length}"
        )));
    }
    Ok(())
}

const MAX_BOS_DURATION_FRAMES: i64 = 16;

fn normalize_boundary_token_durations(mut durations: Vec<i64>) -> Vec<i64> {
    // The Python reference keeps BOS/EOS durations in the alignment. Keeping a
    // bounded BOS duration gives the first real token enough onset context,
    // while suppressing EOS avoids long synthesized tails for short utterances.
    if let Some(first) = durations.first_mut() {
        *first = (*first).min(MAX_BOS_DURATION_FRAMES);
    }
    if durations.len() > 1 {
        let last_idx = durations.len() - 1;
        durations[last_idx] = 0;
    }
    durations
}

#[cfg(test)]
mod tests {
    use super::{
        normalize_boundary_token_durations, validate_context_length, MAX_BOS_DURATION_FRAMES,
    };

    #[test]
    fn context_length_allows_equal_length() {
        assert!(validate_context_length(510, 510).is_ok());
    }

    #[test]
    fn context_length_rejects_overlong_input() {
        let err = validate_context_length(511, 510).unwrap_err();
        assert!(err.to_string().contains("Input too long: 511 > 510"));
    }

    #[test]
    fn boundary_duration_normalization_preserves_capped_bos_and_suppresses_eos() {
        let durations = normalize_boundary_token_durations(vec![4, 7, 9, 3]);
        assert_eq!(durations, vec![4, 7, 9, 0]);

        let capped = normalize_boundary_token_durations(vec![40, 7, 9, 3]);
        assert_eq!(capped, vec![MAX_BOS_DURATION_FRAMES, 7, 9, 0]);
    }

    #[test]
    fn boundary_duration_normalization_caps_single_boundary_duration() {
        assert_eq!(normalize_boundary_token_durations(vec![2]), vec![2]);
        assert_eq!(
            normalize_boundary_token_durations(vec![40]),
            vec![MAX_BOS_DURATION_FRAMES]
        );
        assert_eq!(
            normalize_boundary_token_durations(Vec::new()),
            Vec::<i64>::new()
        );
    }
}
