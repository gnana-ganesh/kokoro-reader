//! Optional pre-G2P text fixes for technical reading (O'Reilly books, docs).
//! Applied before phonemization; only used with G2P backends that understand misaki's
//! `[word](/phonemes/)` markup (loqui, misaki-rs). Disable with KOKORO_TECH_LEXICON=0.

use regex::{Captures, Regex};
use std::sync::OnceLock;

/// word -> misaki-style phonemes. Words where misaki 0.9.4 / loqui-g2p mispronounce tech terms.
const BUILTIN: &[(&str, &str)] = &[
    ("nginx", "ˈɛnʤɪn ˌɛks"),
    ("PostgreSQL", "pˈOstɡɹɛs ˌkjuˈɛl"),
    ("Postgres", "pˈOstɡɹɛs"),
    ("SQLite", "ˌɛskjuˌɛl lˈIt"),
    ("MySQL", "mˌI ˌɛskjuˈɛl"),
    ("PyTorch", "pˈI tˈɔɹʧ"),
    ("NumPy", "nˈʌm pˈI"),
    ("SciPy", "sˈI pˈI"),
    ("kubectl", "kˈub kəntɹˌOl"),
    ("Redis", "ɹˈɛdɪs"),
    ("Kubernetes", "kˌubəɹnˈɛTiz"),
];

const MONTHS: [&str; 12] = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
];

struct Rules {
    iso_date: Regex,
    words: Vec<(Regex, String)>,
}

fn rules() -> &'static Rules {
    static R: OnceLock<Rules> = OnceLock::new();
    R.get_or_init(|| {
        let mut entries: Vec<(String, String)> =
            BUILTIN.iter().map(|(w, p)| (w.to_string(), p.to_string())).collect();
        // User lexicon: KOKORO_LEXICON=/path/file.tsv, lines "word<TAB>phonemes".
        if let Ok(path) = std::env::var("KOKORO_LEXICON") {
            if let Ok(s) = std::fs::read_to_string(&path) {
                for line in s.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')) {
                    if let Some((w, p)) = line.split_once('\t') {
                        entries.retain(|(e, _)| !e.eq_ignore_ascii_case(w.trim()));
                        entries.push((w.trim().to_string(), p.trim().to_string()));
                    }
                }
            } else {
                eprintln!("[kokoro-rs] cannot read KOKORO_LEXICON={path}");
            }
        }
        let words = entries
            .into_iter()
            .map(|(w, p)| {
                let re = Regex::new(&format!(r"(?i)(^|[^\w\]\[]){}($|[^\w\]])", regex::escape(&w))).unwrap();
                (re, p)
            })
            .collect();
        Rules {
            iso_date: Regex::new(r"\b(\d{4})-(\d{2})-(\d{2})\b").unwrap(),
            words,
        }
    })
}

pub fn enabled() -> bool {
    std::env::var("KOKORO_TECH_LEXICON").map(|v| v != "0").unwrap_or(true)
}

pub fn apply(text: &str) -> String {
    let r = rules();
    // 2024-03-15 -> March 15, 2024 (misaki reads ISO dates as digit groups)
    let mut out = r
        .iso_date
        .replace_all(text, |c: &Captures| {
            let m: usize = c[2].parse().unwrap_or(0);
            let d: usize = c[3].parse().unwrap_or(0);
            if (1..=12).contains(&m) && (1..=31).contains(&d) {
                format!("{} {}, {}", MONTHS[m - 1], d, &c[1])
            } else {
                c[0].to_string()
            }
        })
        .into_owned();
    for (re, ps) in &r.words {
        out = re
            .replace_all(&out, |c: &Captures| {
                let word = &c[0][c[1].len()..c[0].len() - c[2].len()];
                format!("{}[{}](/{}/){}", &c[1], word, ps, &c[2])
            })
            .into_owned();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dates_and_words() {
        assert_eq!(apply("On 2024-03-15, use nginx."), "On March 15, 2024, use [nginx](/ˈɛnʤɪn ˌɛks/).");
        assert_eq!(apply("2024-13-40"), "2024-13-40");
        assert_eq!(apply("nginxs and engine"), "nginxs and engine");
        assert_eq!(apply("PostgreSQL, Postgres"), "[PostgreSQL](/pˈOstɡɹɛs ˌkjuˈɛl/), [Postgres](/pˈOstɡɹɛs/)");
    }
}
