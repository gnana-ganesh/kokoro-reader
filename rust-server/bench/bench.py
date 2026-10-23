"""usage: python bench.py URL PID LABEL [runs] -> prints JSON line with latency + RSS"""
import sys, json, time, http.client, statistics
from urllib.parse import urlparse
url, pid, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
runs = int(sys.argv[4]) if len(sys.argv) > 4 else 5
text = open(sys.argv[5] if len(sys.argv) > 5 else "para53.txt").read().strip()
def rss():
    d = {}
    try:
        for l in open(f"/proc/{pid}/status"):
            if l.startswith(("VmRSS", "VmHWM")): d[l.split(":")[0]] = int(l.split()[1]) // 1024
    except FileNotFoundError:  # macOS: no /proc; current RSS via ps, no peak
        import subprocess
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        if out: d["VmRSS"] = int(out) // 1024
    return d
u = urlparse(url)
lat, dur = [], None
for i in range(runs + 1):
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=300)
    t = time.time()
    c.request("POST", "/tts", body=json.dumps({"text": text, "voice": "af_heart", "speed": 1.0}), headers={"Content-Type": "application/json"})
    r = c.getresponse(); b = r.read(); el = time.time() - t
    assert r.status == 200, (r.status, b[:200])
    dur = float(r.getheader("X-Audio-Duration"))
    if i > 0: lat.append(el)   # first run = warm-up
    open(f"out_{label}.wav", "wb").write(b)
m = rss()
print(json.dumps({"label": label, "words": len(text.split()), "audio_s": dur, "median_s": round(statistics.median(lat), 2),
                  "min_s": round(min(lat), 2), "max_s": round(max(lat), 2), "rtf_x": round(dur / statistics.median(lat), 1),
                  "rss_mb": m.get("VmRSS"), "peak_rss_mb": m.get("VmHWM")}))
