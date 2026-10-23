//! kokoro-reader addition: opt-in per-stage timing (KOKORO_PROFILE=1). `lap(label)` charges the
//! wall time since the previous lap (anywhere in the model) to `label`, after synchronizing the
//! device so queued Metal work is attributed to the stage that enqueued it. Off by default: then
//! every call is a cheap no-op and no extra syncs happen.

use candle_core::Device;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static ON: OnceLock<bool> = OnceLock::new();
static STATE: Mutex<(Option<Instant>, Vec<(&'static str, f64)>)> = Mutex::new((None, Vec::new()));

pub fn enabled() -> bool {
    *ON.get_or_init(|| std::env::var("KOKORO_PROFILE").map(|v| v == "1").unwrap_or(false))
}

pub fn begin(dev: &Device) {
    if enabled() {
        let _ = dev.synchronize();
        STATE.lock().unwrap().0 = Some(Instant::now());
    }
}

pub fn lap(dev: &Device, label: &'static str) {
    if !enabled() {
        return;
    }
    let _ = dev.synchronize();
    let mut st = STATE.lock().unwrap();
    if let Some(t) = st.0 {
        let el = t.elapsed().as_secs_f64();
        st.1.push((label, el));
    }
    st.0 = Some(Instant::now());
}

/// Drain recorded laps, summed per label in first-seen order.
pub fn take() -> Vec<(&'static str, f64)> {
    let mut st = STATE.lock().unwrap();
    let mut out: Vec<(&'static str, f64)> = Vec::new();
    for (l, t) in st.1.drain(..) {
        if let Some(e) = out.iter_mut().find(|e| e.0 == l) {
            e.1 += t;
        } else {
            out.push((l, t));
        }
    }
    out
}
