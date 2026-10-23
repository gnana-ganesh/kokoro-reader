"""API parity test: runs the same requests against two servers and diffs status/headers/body.
usage: python parity.py http://127.0.0.1:8880 http://127.0.0.1:8881"""
import sys, json, http.client, io, wave
from urllib.parse import urlparse

EXT = "chrome-extension://" + "abcdefghijklmnopabcdefghijklmnop"
CASES = [
    ("health", "GET", "/health", None, {}),
    ("voices", "GET", "/voices", None, {}),
    ("health ext origin", "GET", "/health", None, {"Origin": EXT}),
    ("health evil origin", "GET", "/health", None, {"Origin": "https://example.com"}),
    ("tts evil origin", "POST", "/tts", {"text": "hi"}, {"Origin": "https://example.com"}),
    ("bad ext id (uppercase)", "GET", "/health", None, {"Origin": "chrome-extension://ABCDEFGHIJKLMNOPABCDEFGHIJKLMNOP"}),
    ("bad ext id (31 chars)", "GET", "/health", None, {"Origin": "chrome-extension://" + "a" * 31}),
    ("null origin", "GET", "/health", None, {"Origin": "null"}),
    ("preflight ok", "OPTIONS", "/tts", None, {"Origin": EXT, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}),
    ("preflight evil", "OPTIONS", "/tts", None, {"Origin": "https://example.com", "Access-Control-Request-Method": "POST"}),
    ("preflight bad method", "OPTIONS", "/tts", None, {"Origin": EXT, "Access-Control-Request-Method": "DELETE"}),
    ("options no origin", "OPTIONS", "/tts", None, {}),
    ("tts ok", "POST", "/tts", {"text": "Hello there.", "voice": "af_heart", "speed": 1.0}, {}),
    ("tts ok ext origin", "POST", "/tts", {"text": "Hello there."}, {"Origin": EXT}),
    ("tts british", "POST", "/tts", {"text": "Hello there.", "voice": "bm_george"}, {}),
    ("tts empty", "POST", "/tts", {"text": "   \u00a0 "}, {}),
    ("tts unknown voice", "POST", "/tts", {"text": "hi", "voice": "zz_nobody"}, {}),
    ("tts speed too high", "POST", "/tts", {"text": "hi", "speed": 3}, {}),
    ("tts speed too low", "POST", "/tts", {"text": "hi", "speed": 0.1}, {}),
    ("tts missing text", "POST", "/tts", {"voice": "af_heart"}, {}),
    ("tts text not string", "POST", "/tts", {"text": 5}, {}),
    ("tts invalid json", "POST", "/tts", b"{not json", {}),
    ("GET /tts", "GET", "/tts", None, {}),
    ("404", "GET", "/nope", None, {}),
]

def req(base, method, path, body, headers):
    u = urlparse(base)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=120)
    h = dict(headers)
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()

INTERESTING = ["content-type", "access-control-allow-origin", "access-control-allow-methods",
               "access-control-allow-headers", "access-control-expose-headers", "access-control-max-age",
               "vary", "cache-control", "access-control-allow-credentials"]

def summarize(status, h, b):
    s = {"status": status}
    for k in INTERESTING:
        if k in h: s[k] = h[k]
    if h.get("content-type", "").startswith("audio/wav"):
        w = wave.open(io.BytesIO(b))
        s["wav"] = dict(ch=w.getnchannels(), rate=w.getframerate(), width=w.getsampwidth())
        s["x-audio-duration?"] = "x-audio-duration" in h; s["x-synth-seconds?"] = "x-synth-seconds" in h
    else:
        try:
            j = json.loads(b)
            if isinstance(j.get("detail"), list):  # pydantic error: compare type/loc/msg only
                j = {"detail": [{k: d.get(k) for k in ("type", "loc", "msg")} for d in j["detail"]]}
            s["body"] = j
        except Exception:
            s["body"] = b[:80].decode(errors="replace")
    return s

a, b = sys.argv[1], sys.argv[2]
fails = 0
for name, m, p, body, hdr in CASES:
    ra, rb = summarize(*req(a, m, p, body, hdr)), summarize(*req(b, m, p, body, hdr))
    # json_invalid loc carries a parser-specific position
    for r in (ra, rb):
        if isinstance(r.get("body"), dict) and isinstance(r["body"].get("detail"), list):
            for d in r["body"]["detail"]:
                if d.get("type") == "json_invalid": d["loc"] = ["body", "<pos>"]
    ok = ra == rb
    fails += not ok
    print(("PASS" if ok else "DIFF"), name, "->", ra["status"])
    if not ok:
        print("   py:", ra); print("   rs:", rb)
print(f"\n{len(CASES)-fails}/{len(CASES)} identical")
