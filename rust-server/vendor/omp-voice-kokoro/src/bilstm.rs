//! Bidirectional LSTM for candle.
//!
//! kokoro-reader change: the upstream version ran candle's LSTM step by step on the model's
//! device and reversed the sequence with one slice per timestep. On Metal that is thousands of
//! tiny kernel dispatches per utterance (Kokoro runs ~5 BiLSTMs over tokens and one over
//! frames). This version always runs the recurrence on the CPU: one matmul for the input
//! projection of all timesteps, then a tight f32 matvec loop per step, forward and backward
//! directions on two threads. Set KOKORO_LSTM=candle to use the original implementation.

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{self as nn, LSTMConfig, RNN, VarBuilder, rnn::Direction};

struct Dir {
    /// W_ih^T on CPU: [in, 4h]
    w_ih_t: Tensor,
    /// b_ih + b_hh: [4h]
    bias: Tensor,
    /// W_hh row-major [4h][h]
    w_hh: Vec<f32>,
}

pub struct BiLSTM {
    fwd: nn::LSTM,
    bwd: nn::LSTM,
    cpu: [Dir; 2],
    hidden: usize,
    use_candle: bool,
}

fn load_dir(in_dim: usize, h: usize, vb: &VarBuilder, sfx: &str) -> Result<Dir> {
    let g = |name: &str, s: (usize, usize)| vb.get(s, &format!("{name}{sfx}"))?.to_device(&Device::Cpu)?.to_dtype(DType::F32);
    let w_ih = g("weight_ih_l0", (4 * h, in_dim))?;
    let w_hh = g("weight_hh_l0", (4 * h, h))?;
    let b_ih = vb.get(4 * h, &format!("bias_ih_l0{sfx}"))?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let b_hh = vb.get(4 * h, &format!("bias_hh_l0{sfx}"))?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    Ok(Dir {
        w_ih_t: w_ih.t()?.contiguous()?,
        bias: (b_ih + b_hh)?,
        w_hh: w_hh.flatten_all()?.to_vec1::<f32>()?,
    })
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// gx: [T][4h] precomputed input projection (+biases). Returns [T][h] hidden states.
fn run_dir(gx: &[f32], t_len: usize, h: usize, w_hh: &[f32], reverse: bool) -> Vec<f32> {
    let mut out = vec![0f32; t_len * h];
    let mut hs = vec![0f32; h];
    let mut cs = vec![0f32; h];
    let mut gates = vec![0f32; 4 * h];
    for step in 0..t_len {
        let t = if reverse { t_len - 1 - step } else { step };
        let gxt = &gx[t * 4 * h..(t + 1) * 4 * h];
        for r in 0..4 * h {
            gates[r] = gxt[r] + dot(&w_hh[r * h..(r + 1) * h], &hs);
        }
        // PyTorch gate order: input, forget, cell, output
        for j in 0..h {
            let i = sigmoid(gates[j]);
            let f = sigmoid(gates[h + j]);
            let g = gates[2 * h + j].tanh();
            let o = sigmoid(gates[3 * h + j]);
            cs[j] = f * cs[j] + i * g;
            hs[j] = o * cs[j].tanh();
        }
        out[t * h..(t + 1) * h].copy_from_slice(&hs);
    }
    out
}

impl BiLSTM {
    pub fn load(in_dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let fwd_cfg = LSTMConfig { direction: Direction::Forward, ..Default::default() };
        let bwd_cfg = LSTMConfig { direction: Direction::Backward, ..Default::default() };
        let fwd = nn::lstm(in_dim, hidden_dim, fwd_cfg, vb.clone())?;
        let bwd = nn::lstm(in_dim, hidden_dim, bwd_cfg, vb.clone())?;
        let cpu = [load_dir(in_dim, hidden_dim, &vb, "")?, load_dir(in_dim, hidden_dim, &vb, "_reverse")?];
        let use_candle = std::env::var("KOKORO_LSTM").map(|v| v == "candle").unwrap_or(false);
        Ok(Self { fwd, bwd, cpu, hidden: hidden_dim, use_candle })
    }

    /// input: [B, T, features] -> [B, T, 2*hidden_dim]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (b, seq_len, _feat) = input.dims3()?;
        if self.use_candle || b != 1 {
            return self.forward_candle(input);
        }
        let h = self.hidden;
        let x = input.squeeze(0)?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?; // [T, F]
        let gx: Vec<Vec<f32>> = self
            .cpu
            .iter()
            .map(|d| x.matmul(&d.w_ih_t)?.broadcast_add(&d.bias)?.flatten_all()?.to_vec1::<f32>())
            .collect::<Result<_>>()?;
        let (of, ob) = std::thread::scope(|s| {
            let jf = s.spawn(|| run_dir(&gx[0], seq_len, h, &self.cpu[0].w_hh, false));
            let ob = run_dir(&gx[1], seq_len, h, &self.cpu[1].w_hh, true);
            (jf.join().expect("lstm thread"), ob)
        });
        let mut out = Vec::with_capacity(seq_len * 2 * h);
        for t in 0..seq_len {
            out.extend_from_slice(&of[t * h..(t + 1) * h]);
            out.extend_from_slice(&ob[t * h..(t + 1) * h]);
        }
        Tensor::from_vec(out, (1, seq_len, 2 * h), input.device())?.to_dtype(input.dtype())
    }

    fn forward_candle(&self, input: &Tensor) -> Result<Tensor> {
        let (_b, seq_len, _feat) = input.dims3()?;
        let fwd_states = self.fwd.seq(input)?;
        let fwd_h: Vec<Tensor> = fwd_states.iter().map(|s| s.h().clone()).collect();
        let fwd_out = Tensor::stack(&fwd_h, 1)?;
        let mut bwd_slices = Vec::with_capacity(seq_len);
        for i in (0..seq_len).rev() {
            bwd_slices.push(input.i((.., i..i + 1, ..))?.clone());
        }
        let reversed = Tensor::cat(&bwd_slices, 1)?;
        let bwd_states = self.bwd.seq(&reversed)?;
        let bwd_h: Vec<Tensor> = bwd_states.iter().map(|s| s.h().clone()).collect();
        let bwd_out_reversed = Tensor::stack(&bwd_h, 1)?;
        let mut bwd_slices_out = Vec::with_capacity(seq_len);
        for i in (0..seq_len).rev() {
            bwd_slices_out.push(bwd_out_reversed.i((.., i..i + 1, ..))?.clone());
        }
        let bwd_out = Tensor::cat(&bwd_slices_out, 1)?;
        Tensor::cat(&[&fwd_out, &bwd_out], 2)
    }
}
