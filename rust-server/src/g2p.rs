//! Grapheme-to-phoneme front-ends. Kokoro-82M v1.0 was trained on *misaki* phonemes
//! (e.g. `A` = eɪ, `O` = oʊ, `ᵊ`, `ɾ`), so a misaki-compatible G2P matters for quality.
//!
//! * `loqui`    – loqui-g2p (MIT): Rust port of misaki 0.9.4 incl. spaCy-style tokenizer,
//!                POS tagger, gold/silver lexicons, numbers; neural OOV fallback. No GPL code.
//! * `misaki-rs`– misaki-rs (MIT): another misaki port; OOV fallback via espeak-ng (GPL-3.0).
//! * `espeak`   – raw espeak-ng IPA, which is what Kokoros (lucasjinreal/Kokoros) feeds the model.

#[allow(unused_imports)]
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lang {
    American,
    British,
}

impl Lang {
    pub fn for_voice(voice: &str) -> Lang {
        if voice.starts_with('b') { Lang::British } else { Lang::American }
    }
}

pub trait Phonemizer: Send + Sync {
    fn name(&self) -> &'static str;
    fn phonemize(&self, text: &str, lang: Lang) -> Result<String, String>;
    /// Understands misaki's `[word](/phonemes/)` inline markup.
    fn supports_markup(&self) -> bool {
        false
    }
}

pub fn available() -> Vec<&'static str> {
    let mut v = Vec::new();
    #[cfg(feature = "g2p-loqui")]
    v.push("loqui");
    #[cfg(feature = "g2p-misaki-rs")]
    v.push("misaki-rs");
    #[cfg(feature = "g2p-espeak")]
    v.push("espeak");
    v
}

pub fn build(name: &str) -> Result<Box<dyn Phonemizer>, String> {
    match name {
        #[cfg(feature = "g2p-loqui")]
        "loqui" => Ok(Box::new(Loqui::new()?)),
        #[cfg(feature = "g2p-misaki-rs")]
        "misaki-rs" | "misaki" => Ok(Box::new(MisakiRs::new())),
        #[cfg(feature = "g2p-espeak")]
        "espeak" => Ok(Box::new(Espeak)),
        other => Err(format!(
            "G2P backend {other:?} is not compiled in (available: {:?})",
            available()
        )),
    }
}

// ---------------------------------------------------------------- loqui-g2p
#[cfg(feature = "g2p-loqui")]
pub struct Loqui {
    us: loqui_g2p::G2p,
    gb: loqui_g2p::G2p,
}

#[cfg(feature = "g2p-loqui")]
impl Loqui {
    pub fn new() -> Result<Self, String> {
        use loqui_g2p::{Dialect, G2p, OovFallback};
        Ok(Self {
            us: G2p::new(Dialect::American, OovFallback::Neural).map_err(|e| e.to_string())?,
            gb: G2p::new(Dialect::British, OovFallback::Neural).map_err(|e| e.to_string())?,
        })
    }
}

#[cfg(feature = "g2p-loqui")]
impl Phonemizer for Loqui {
    fn name(&self) -> &'static str { "loqui" }
    fn supports_markup(&self) -> bool { true }
    fn phonemize(&self, text: &str, lang: Lang) -> Result<String, String> {
        let g = if lang == Lang::British { &self.gb } else { &self.us };
        Ok(g.phonemize(text))
    }
}

// ---------------------------------------------------------------- misaki-rs
#[cfg(feature = "g2p-misaki-rs")]
pub struct MisakiRs {
    us: Mutex<misaki_rs::G2P>,
    gb: Mutex<misaki_rs::G2P>,
}

#[cfg(feature = "g2p-misaki-rs")]
impl MisakiRs {
    pub fn new() -> Self {
        Self {
            us: Mutex::new(misaki_rs::G2P::new(misaki_rs::Language::EnglishUS)),
            gb: Mutex::new(misaki_rs::G2P::new(misaki_rs::Language::EnglishGB)),
        }
    }
}

#[cfg(feature = "g2p-misaki-rs")]
impl Phonemizer for MisakiRs {
    fn name(&self) -> &'static str { "misaki-rs" }
    fn phonemize(&self, text: &str, lang: Lang) -> Result<String, String> {
        let g = if lang == Lang::British { &self.gb } else { &self.us };
        let g = g.lock().map_err(|e| e.to_string())?;
        g.g2p(text).map(|(ps, _)| ps).map_err(|e| format!("{e:?}"))
    }
}

// ---------------------------------------------------------------- espeak-ng
#[cfg(feature = "g2p-espeak")]
pub struct Espeak;

#[cfg(feature = "g2p-espeak")]
static ESPEAK_LOCK: Mutex<()> = Mutex::new(());

#[cfg(feature = "g2p-espeak")]
impl Phonemizer for Espeak {
    fn name(&self) -> &'static str { "espeak" }
    fn phonemize(&self, text: &str, lang: Lang) -> Result<String, String> {
        // Same call Kokoros makes (koko.rs: text_to_phonemes(text, lan, None, true, false)).
        let id = if lang == Lang::British { "en" } else { "en-us" };
        let _g = ESPEAK_LOCK.lock().map_err(|e| e.to_string())?;
        espeak_rs::text_to_phonemes(text, id, None, true, false)
            .map(|v| v.join(""))
            .map_err(|e| format!("{e:?}"))
    }
}

