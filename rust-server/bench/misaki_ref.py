import sys, warnings; warnings.filterwarnings("ignore")
from misaki import en, espeak
lang = sys.argv[1] if len(sys.argv)>1 else 'a'
g2p = en.G2P(trf=False, british=(lang=='b'), fallback=espeak.EspeakFallback(british=(lang=='b')))
for line in sys.stdin:
    line=line.strip()
    if line: print(g2p(line)[0])
