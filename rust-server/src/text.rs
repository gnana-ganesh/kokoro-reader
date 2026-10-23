//! Text clean-up and sentence-level chunking.

/// Same as server.py `_clean`: NBSP -> space, collapse whitespace, trim, cap at 6000 chars.
pub const MAX_CHARS: usize = 6000;

pub fn clean(text: &str) -> String {
    let t = text.replace('\u{00a0}', " ");
    let collapsed = t.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed.chars().take(MAX_CHARS).collect()
}

const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "cf", "fig",
    "figs", "no", "nos", "vol", "vols", "ch", "sec", "eq", "approx", "inc", "ltd", "co", "corp",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "u.s",
    "u.k", "a.m", "p.m", "al", "ed", "eds", "rev", "gen", "gov", "lt", "col", "capt", "mt", "ft",
];

fn is_abbreviation(word: &str) -> bool {
    let w = word
        .trim_start_matches(|c: char| !c.is_alphanumeric())
        .trim_end_matches('.')
        .to_lowercase();
    if w.is_empty() {
        return false;
    }
    // Single letters ("J. R. R. Tolkien", "A. Smith") and dotted initialisms ("U.S.A.").
    if w.chars().count() == 1 || (w.contains('.') && w.split('.').all(|p| p.chars().count() <= 2)) {
        return true;
    }
    ABBREVIATIONS.contains(&w.as_str())
}

/// Split into sentences. A boundary is `.`, `!`, `?` or `…` (optionally followed by closing
/// quotes/brackets) followed by whitespace. Decimal numbers (`3.14`), versions (`v1.2.3`),
/// `std::vector`, URLs and abbreviations (`e.g.`, `Dr.`) are never split. Sentence-final
/// punctuation is kept so the model gets question/exclamation intonation.
pub fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if matches!(c, '.' | '!' | '?' | '…') {
            let mut j = i + 1;
            while j < chars.len() && matches!(chars[j], '.' | '!' | '?' | '…') {
                j += 1;
            }
            while j < chars.len() && matches!(chars[j], '"' | '\'' | '”' | '’' | ')' | ']' | '»') {
                j += 1;
            }
            if j < chars.len() && chars[j].is_whitespace() {
                let mut boundary = true;
                if c == '.' && j == i + 1 {
                    // the word that ends with this period
                    let mut k = i;
                    while k > start && !chars[k - 1].is_whitespace() {
                        k -= 1;
                    }
                    let word: String = chars[k..=i].iter().collect();
                    let mut n = j;
                    while n < chars.len() && chars[n].is_whitespace() {
                        n += 1;
                    }
                    let next_lower = n < chars.len() && chars[n].is_lowercase();
                    if is_abbreviation(&word) || next_lower {
                        boundary = false;
                    }
                }
                if boundary {
                    let s: String = chars[start..j].iter().collect();
                    let s = s.trim().to_string();
                    if !s.is_empty() {
                        out.push(s);
                    }
                    start = j;
                }
            }
            i = j;
            continue;
        }
        i += 1;
    }
    let rest: String = chars[start..].iter().collect();
    let rest = rest.trim().to_string();
    if !rest.is_empty() {
        out.push(rest);
    }
    out
}

/// Split one over-long sentence at clause punctuation, then at word boundaries,
/// into pieces of at most `max_chars` characters.
pub fn split_long(sentence: &str, max_chars: usize) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut cur = String::new();
    for word in sentence.split(' ') {
        let candidate_len = cur.chars().count() + 1 + word.chars().count();
        let clause_break = cur.ends_with(',') || cur.ends_with(';') || cur.ends_with(':') || cur.ends_with('—');
        if !cur.is_empty() && (candidate_len > max_chars || (clause_break && cur.chars().count() > max_chars / 2)) {
            pieces.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
    }
    if !cur.is_empty() {
        pieces.push(cur);
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_matches_python() {
        assert_eq!(clean("  a\u{a0}b\n\n c\t"), "a b c");
        assert_eq!(clean(&"x".repeat(7000)).len(), 6000);
    }

    #[test]
    fn sentences() {
        let s = split_sentences(
            "Dr. Smith paid $3.50 for v1.2.3, e.g. at 9 a.m. today. Is std::vector fast? Yes! See https://x.org/a.b now.",
        );
        assert_eq!(
            s,
            vec![
                "Dr. Smith paid $3.50 for v1.2.3, e.g. at 9 a.m. today.",
                "Is std::vector fast?",
                "Yes!",
                "See https://x.org/a.b now."
            ]
        );
        let s = split_sentences("Height: 8,848.86 meters (29,031.7 feet) above sea level. Next one.");
        assert_eq!(s.len(), 2);
        assert_eq!(s[0], "Height: 8,848.86 meters (29,031.7 feet) above sea level.");
        assert_eq!(split_sentences("He said \"Stop.\" Then he left."), vec!["He said \"Stop.\"", "Then he left."]);
    }

    #[test]
    fn long_split() {
        let s = "word ".repeat(100);
        for p in split_long(s.trim(), 50) {
            assert!(p.chars().count() <= 50);
        }
    }
}
