"""Compare WAVs against a reference (stdlib only): duration ratio, RMS, silent fraction, and
envelope correlation (20 ms RMS frames) - catches silence, noise, truncation, or wrong speech.
usage: python3 audio_cmp.py REF.wav OTHER.wav [...]"""
import sys, wave, array, math, json
def load(p):
    w = wave.open(p); return w.getframerate(), array.array('h', w.readframes(w.getnframes()))
def env(a, win):
    return [math.sqrt(sum(x*x for x in a[i:i+win])/win) for i in range(0, len(a)-win, win)]
def corr(x, y):
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    mx, my = sum(x)/n, sum(y)/n
    sxy = sum((a-mx)*(b-my) for a, b in zip(x, y)); sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    return sxy/(sx*sy) if sx and sy else 0.0
sr, ref = load(sys.argv[1]); win = sr//50; eref = env(ref, win)
for p in sys.argv[1:]:
    sr2, a = load(p); e = env(a, win)
    rms = math.sqrt(sum(x*x for x in a)/max(len(a),1))/32768
    silent = sum(1 for v in e if v < 0.003*32768)/max(len(e),1)
    print(json.dumps({"file": p, "secs": round(len(a)/sr2, 2), "len_vs_ref": round(len(a)/len(ref), 3), "rms": round(rms, 4),
                      "silent_frac": round(silent, 3), "env_corr_vs_ref": round(corr(eref, e), 3)}))
