// Kokoro Reader: floating read-aloud panel.
// Auto-loaded on learning.oreilly.com; injected on demand (toolbar icon / Alt+Shift+P /
// context menu) on any other page.
(() => {
  if (window.__kokoroReaderLoaded) return;
  window.__kokoroReaderLoaded = true;

  const IS_OREILLY = location.hostname === "learning.oreilly.com";

  // ---------- config ----------
  const OREILLY_SELECTORS = ["#sbo-rt-content", "[data-testid='contentViewer']", "#book-content"];
  const CONTAINER_SELECTORS = [
    "main article", "article", "[role='main']", "main",
    "#mw-content-text", "#content", ".post-content", ".entry-content", ".article-body", ".markdown-body",
  ];
  const BLOCK_SELECTOR = "h1,h2,h3,h4,h5,h6,p,li,dt,dd,blockquote,figcaption,caption,th,td,pre,div.note,div.tip,div.warning,aside";
  const CODE_SELECTOR = "pre,[data-type='programlisting'],.programlisting,code.block,table.code";
  const NOISE_SELECTOR = "nav,header,footer,form,[role='navigation'],[role='banner'],[role='contentinfo'],[role='complementary'],[aria-hidden='true'],.sidebar,#sidebar,.comments,#comments,.cookie,.advert,.ad,.ads";
  const STRIP_SELECTOR = "sup a, a[data-type='noteref'], sup.reference, .mw-editsection, .kokoro-skip, script, style, noscript, button, svg, template";
  const FALLBACK_VOICES = [
    { id: "af_heart", label: "Heart (US female)" }, { id: "af_bella", label: "Bella (US female)" },
    { id: "am_michael", label: "Michael (US male)" }, { id: "bf_emma", label: "Emma (UK female)" },
    { id: "bm_george", label: "George (UK male)" },
  ];
  const DEFAULTS = { voice: "af_heart", speed: 1.0, skipCode: true, collapsed: false, hidden: false };

  // ---------- state ----------
  let settings = { ...DEFAULTS };
  let blocks = [];          // [{el, text}]
  let mode = "page";        // page | selection
  let idx = 0;
  let state = "idle";       // idle | loading | playing | paused
  let gen = 0;              // bumps on every stop/jump to cancel stale async work
  let hasPosition = false;
  let panelHidden = IS_OREILLY ? false : true; // non-O'Reilly: shown when injected on request
  let lastDetection = "";
  const cache = new Map();  // key -> Promise<AudioBuffer>

  // Audio goes through Web Audio (decoded buffers), not <audio src=blob:>, so sites with a
  // strict Content-Security-Policy (media-src) can't block playback.
  let actx = null;
  let source = null;        // current AudioBufferSourceNode
  const ctx = () => (actx || (actx = new (window.AudioContext || window.webkitAudioContext)()));
  function stopSource() {
    const s = source;
    source = null;
    if (s) { try { s.onended = null; s.stop(); } catch (_) {} try { s.disconnect(); } catch (_) {} }
  }

  // Remove a stale panel left behind by a previous (reloaded) version of the extension.
  document.querySelectorAll("#kokoro-reader-host").forEach((n) => n.remove());

  // ---------- helpers ----------
  const send = (msg) => new Promise((resolve) => {
    try {
      chrome.runtime.sendMessage(msg, (resp) => {
        if (chrome.runtime.lastError) resolve({ ok: false, error: chrome.runtime.lastError.message, offline: true });
        else resolve(resp || { ok: false, error: "no response" });
      });
    } catch (e) { resolve({ ok: false, error: String(e), offline: true }); }
  });

  const norm = (s) => (s || "").replace(/\s+/g, " ").trim();
  // Matching key between Readability's cleaned copy and the live DOM.
  const keyOf = (s) => (s || "").toLowerCase().replace(/[^a-z0-9\u00c0-\uffff]+/g, "").slice(0, 80);

  function isVisible(el) {
    if (!el.isConnected) return false;
    const st = getComputedStyle(el);
    return st.display !== "none" && st.visibility !== "hidden" && el.getClientRects().length > 0;
  }

  function ownText(el) {
    // Text of el, minus nested blocks (they are read separately) and footnote markers.
    const clone = el.cloneNode(true);
    clone.querySelectorAll(BLOCK_SELECTOR).forEach((n) => n.remove());
    clone.querySelectorAll(STRIP_SELECTOR).forEach((n) => n.remove());
    clone.querySelectorAll("img[alt]").forEach((n) => n.replaceWith(" "));
    return norm(clone.textContent);
  }

  const inHost = (el) => host.contains(el);
  const isCode = (el) => el.matches(CODE_SELECTOR);

  // Candidate live elements: the usual block tags plus "leaf" divs that hold text directly
  // (many sites put paragraphs in <div>s).
  function candidates(root, generic) {
    const list = [...root.querySelectorAll(BLOCK_SELECTOR)];
    if (generic) {
      for (const d of root.querySelectorAll("div,section")) {
        if (d.querySelector("div,section,article," + BLOCK_SELECTOR)) continue;
        let direct = 0;
        for (const n of d.childNodes) if (n.nodeType === 3 || (n.nodeType === 1 && /^(A|SPAN|EM|STRONG|B|I|CODE|MARK|SMALL|SUB|SUP|ABBR|TIME|Q|CITE|BR)$/.test(n.tagName))) direct += (n.textContent || "").trim().length;
        if (direct >= 40) list.push(d);
      }
      // keep document order
      list.sort((a, b) => (a === b ? 0 : a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1));
    }
    return list;
  }

  function collect(root, { generic = false, filter = null, dropNoise = false } = {}) {
    const out = [];
    for (const el of candidates(root, generic)) {
      if (inHost(el) || !isVisible(el)) continue;
      const code = isCode(el) || !!el.closest(CODE_SELECTOR);
      if (code && settings.skipCode) continue;
      if (!isCode(el) && el.closest("pre")) continue; // never split inside code
      if (dropNoise && el.closest(NOISE_SELECTOR)) continue;
      const text = isCode(el) ? norm(el.textContent) : ownText(el);
      if (!text || !/[A-Za-z0-9\u00c0-\uffff]/.test(text)) continue;
      if (filter && !filter(el, text)) continue;
      out.push({ el, text });
    }
    return out;
  }

  // 1) O'Reilly reader container.
  function scanOReilly() {
    for (const sel of OREILLY_SELECTORS) {
      for (const el of document.querySelectorAll(sel)) {
        if (!inHost(el) && (el.innerText || "").trim().length > 200) return collect(el);
      }
    }
    return null;
  }

  // 2) Mozilla Readability on a copy of the page, mapped back to live elements by text.
  function scanReadability() {
    if (typeof Readability !== "function") return null;
    let article;
    try {
      const copy = document.cloneNode(true);
      copy.querySelectorAll("#kokoro-reader-host").forEach((n) => n.remove());
      article = new Readability(copy, { serializer: (el) => el, charThreshold: 300 }).parse();
    } catch (e) { return null; }
    if (!article || !article.content) return null;
    const keys = new Set();
    const addKey = (s) => { const k = keyOf(s); if (k.length >= 4) keys.add(k); };
    for (const el of article.content.querySelectorAll(BLOCK_SELECTOR + ",div,section")) {
      addKey(ownText(el));
      addKey(el.textContent);
    }
    if (article.title) { addKey(article.title); addKey(article.title.split(/\s[-|–—:]\s/)[0]); }
    if (!keys.size) return null;
    let found = collect(document.body, {
      generic: true,
      filter: (el, text) => keys.has(keyOf(text)) && !el.closest("nav,[role='navigation'],.toc,#toc"),
    });
    // Keep only the article region: the deepest live ancestor holding >= 95% of the matched
    // text (drops tables of contents, related-links boxes, etc. that repeat headings).
    const total = found.reduce((n, b) => n + b.text.length, 0);
    const weight = new Map();
    for (const b of found) {
      for (let a = b.el.parentElement; a; a = a.parentElement) weight.set(a, (weight.get(a) || 0) + b.text.length);
    }
    let region = document.body, depth = -1;
    for (const [a, w] of weight) {
      if (w < total * 0.95) continue;
      let d = 0; for (let x = a; x; x = x.parentElement) d++;
      if (d > depth) { depth = d; region = a; }
    }
    found = found.filter((b) => region.contains(b.el));
    // drop exact duplicates where a parent and a child produce the same text
    const out = found.filter((b, i) => !found.some((o, j) => j !== i && o.el !== b.el && b.el.contains(o.el) && o.text === b.text));
    const chars = out.reduce((n, b) => n + b.text.length, 0);
    return out.length >= 3 || chars >= 300 ? out : null;
  }

  // 3) Common article containers.
  function scanContainers() {
    for (const sel of CONTAINER_SELECTORS) {
      for (const el of document.querySelectorAll(sel)) {
        if (inHost(el) || (el.innerText || "").trim().length < 400) continue;
        const out = collect(el, { generic: true });
        if (out.length >= 2) return out;
      }
    }
    return null;
  }

  // 4) Any visible block text, minus nav/header/footer and tiny fragments.
  function scanBody() {
    return collect(document.body, {
      generic: true, dropNoise: true,
      filter: (el, text) => /^H[1-6]$/.test(el.tagName) || isCode(el) || text.length >= 25,
    });
  }

  function scan() {
    mode = "page";
    let out = null;
    if (IS_OREILLY) { out = scanOReilly(); lastDetection = "O'Reilly book"; }
    if (!out || !out.length) { out = scanReadability(); lastDetection = "article"; }
    if (!out || !out.length) { out = scanContainers(); lastDetection = "main content"; }
    if (!out || !out.length) { out = scanBody(); lastDetection = "page text"; }
    blocks = [];
    for (const b of out || []) {
      if (b.text.length <= 1200 || isCode(b.el)) blocks.push(b);
      else for (const c of chunks(b.text)) blocks.push({ el: b.el, text: c });
    }
    return blocks;
  }

  // ---------- selection ----------
  function currentSelection() {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) return null;
    const range = sel.getRangeAt(0);
    const anchor = range.commonAncestorContainer;
    if (inHost(anchor.nodeType === 1 ? anchor : anchor.parentElement)) return null;
    if (!norm(sel.toString())) return null;
    return range;
  }

  // Split long text into sentence groups (~400 chars) so audio starts quickly.
  function chunks(text) {
    const parts = norm(text).match(/[^.!?]+(?:[.!?]+["')\]]*\s*|$)/g) || [text];
    const out = [];
    let cur = "";
    for (const p of parts) {
      if ((cur + p).length > 400 && cur) { out.push(cur.trim()); cur = ""; }
      cur += p;
    }
    if (cur.trim()) out.push(cur.trim());
    return out;
  }

  function selectionBlocks(range, fallbackText) {
    const out = [];
    if (range) {
      const root = range.commonAncestorContainer.nodeType === 1 ? range.commonAncestorContainer : range.commonAncestorContainer.parentElement;
      const scope = root.closest(BLOCK_SELECTOR) || root;
      const els = [scope, ...scope.querySelectorAll(BLOCK_SELECTOR)].filter((el) => el.matches(BLOCK_SELECTOR));
      for (const el of els) {
        if (inHost(el) || !range.intersectsNode(el)) continue;
        if ([...el.querySelectorAll(BLOCK_SELECTOR)].some((c) => range.intersectsNode(c))) continue; // leaf blocks only
        const r = document.createRange();
        r.selectNodeContents(el);
        if (r.compareBoundaryPoints(Range.START_TO_START, range) < 0) r.setStart(range.startContainer, range.startOffset);
        if (r.compareBoundaryPoints(Range.END_TO_END, range) > 0) r.setEnd(range.endContainer, range.endOffset);
        const t = norm(r.toString());
        if (t && /[A-Za-z0-9\u00c0-\uffff]/.test(t)) for (const c of chunks(t)) out.push({ el, text: c });
      }
      if (!out.length) {
        const t = norm(range.toString());
        for (const c of chunks(t)) out.push({ el: root, text: c });
      }
    } else if (fallbackText) {
      for (const c of chunks(fallbackText)) out.push({ el: null, text: c });
    }
    return out;
  }

  function readSelection(range, fallbackText) {
    const list = selectionBlocks(range, fallbackText);
    if (!list.length) return false;
    stop(false);
    blocks = list; mode = "selection"; hasPosition = true;
    showPanel();
    playFrom(0);
    return true;
  }

  function blocksStale() {
    if (mode === "selection") return !blocks.length;
    return !blocks.length || !blocks[0].el.isConnected || !blocks[blocks.length - 1].el.isConnected;
  }

  function firstVisibleIndex() {
    const h = window.innerHeight;
    for (let i = 0; i < blocks.length; i++) {
      const r = blocks[i].el.getBoundingClientRect();
      if (r.bottom > 60 && r.top < h) return i;
    }
    return 0;
  }

  const keyFor = (i) => `${settings.voice}|${settings.speed}|${blocks[i] ? blocks[i].text : ""}`;

  function fetchAudio(i) {
    const key = keyFor(i);
    if (cache.has(key)) return cache.get(key);
    const p = send({ type: "tts", text: blocks[i].text, voice: settings.voice, speed: settings.speed }).then((r) => {
      if (!r.ok) { cache.delete(key); throw r; }
      const bin = atob(r.b64);
      const bytes = new Uint8Array(bin.length);
      for (let k = 0; k < bin.length; k++) bytes[k] = bin.charCodeAt(k);
      return ctx().decodeAudioData(bytes.buffer);
    }).catch((e) => { cache.delete(key); throw e; });
    cache.set(key, p);
    // keep the cache small (current + a few neighbours)
    while (cache.size > 6) {
      const oldKey = cache.keys().next().value;
      const old = cache.get(oldKey);
      cache.delete(oldKey);
      old.catch(() => {});
    }
    return p;
  }

  function clearCache() {
    for (const p of cache.values()) p.catch(() => {});
    cache.clear();
  }

  function clearHighlight() {
    document.querySelectorAll(".kokoro-current").forEach((n) => n.classList.remove("kokoro-current"));
  }

  let lastHighlighted = null;
  function highlight(i) {
    const el = blocks[i] && blocks[i].el;
    if (el === lastHighlighted && el && el.classList.contains("kokoro-current")) return;
    clearHighlight();
    lastHighlighted = el;
    if (!el || el === document.body || el === document.documentElement) return;
    el.classList.add("kokoro-current");
    el.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  // ---------- playback ----------
  async function playFrom(i) {
    const my = ++gen;
    stopSource();
    if (blocksStale()) scan();
    if (!blocks.length) { setStatus("No readable text found on this page."); setState("idle"); return; }
    if (i >= blocks.length) {
      const wasSel = mode === "selection";
      stop(false);
      setStatus(wasSel ? "Finished reading the selection." : "Reached the end of this page.");
      if (wasSel) { blocks = []; mode = "page"; hasPosition = false; clearHighlight(); }
      return;
    }
    idx = Math.max(0, i);
    hasPosition = true;
    highlight(idx);
    setState("loading");
    setStatus(`Synthesizing ${idx + 1}/${blocks.length}…`);
    let buf;
    try {
      buf = await fetchAudio(idx);
    } catch (r) {
      if (my !== gen) return;
      setState("idle");
      setStatus(r && r.offline
        ? "Can't reach the TTS server. Run ~/kokoro-tts/start.sh"
        : `TTS error: ${(r && r.error) || r}`);
      return;
    }
    if (my !== gen) return;
    if (idx + 1 < blocks.length) fetchAudio(idx + 1).catch(() => {});   // prefetch next paragraph
    const c = ctx();
    try { if (c.state !== "running") await c.resume(); } catch (_) {}
    if (my !== gen) return;
    const src = c.createBufferSource();
    src.buffer = buf;
    src.connect(c.destination);
    src.onended = () => { if (source === src && state === "playing") { source = null; playFrom(idx + 1); } };
    source = src;
    src.start();
    if (c.state !== "running") {   // browser autoplay policy: needs a click on the page first
      setState("paused"); setStatus("Press Play to start audio."); return;
    }
    setState("playing");
    setStatus(`Reading ${mode === "selection" ? "selection " : ""}${idx + 1}/${blocks.length}`);
  }

  function togglePlay(selRange) {
    if (state === "playing") { if (actx) actx.suspend(); setState("paused"); setStatus(`Paused at ${idx + 1}/${blocks.length}`); return; }
    if (state === "paused" && source && actx) {
      setState("playing");
      actx.resume().then(() => {
        if (actx.state === "running") setStatus(`Reading ${idx + 1}/${blocks.length}`);
        else { setState("paused"); setStatus("Press Play to start audio."); }
      }).catch(() => {});
      return;
    }
    if (state === "loading") return;
    // A text selection wins: read just that.
    const range = selRange || currentSelection();
    if (range && readSelection(range)) return;
    if (blocksStale()) scan();
    const start = (state === "paused" || hasPosition) ? idx : firstVisibleIndex();
    playFrom(start);
  }

  function stop(clear = true) {
    gen++;
    stopSource();
    if (actx && actx.state === "suspended") actx.resume().catch(() => {});
    setState("idle");
    if (clear) {
      clearHighlight();
      lastHighlighted = null;
      hasPosition = false;
      if (mode === "selection") { blocks = []; mode = "page"; }
      setStatus("Stopped. Play starts at the first visible paragraph (or your selection); Option+click a paragraph to read from there.");
    }
  }

  function skip(delta) {
    if (blocksStale()) scan();
    if (!blocks.length) return;
    const target = Math.min(Math.max(idx + delta, 0), blocks.length - 1);
    if (state === "playing" || state === "loading") playFrom(target);
    else { idx = target; hasPosition = true; highlight(idx); setState("paused"); setStatus(`At ${idx + 1}/${blocks.length}; press Play`); stopSource(); }
  }

  // ---------- Alt(Option)+click "read from here" ----------
  document.addEventListener("click", (e) => {
    if (!e.altKey || inHost(e.target)) return;
    if (mode === "selection" || blocksStale()) scan();
    let el = e.target instanceof Element ? e.target : e.target.parentElement;
    while (el && el !== document.body) {
      const i = blocks.findIndex((b) => b.el === el);
      if (i >= 0) { e.preventDefault(); e.stopPropagation(); hasPosition = true; showPanel(); playFrom(i); return; }
      el = el.parentElement;
    }
  }, true);
  let hoverEl = null;
  let lastHoverScan = 0;
  document.addEventListener("mousemove", (e) => {
    if (!e.altKey) { if (hoverEl) { hoverEl.classList.remove("kokoro-alt-hover"); hoverEl = null; } return; }
    const t = e.target instanceof Element ? e.target.closest(BLOCK_SELECTOR + ",div") : null;
    if (t === hoverEl) return;
    if (hoverEl) hoverEl.classList.remove("kokoro-alt-hover");
    hoverEl = null;
    if (t && !inHost(t) && mode !== "selection") {
      if (blocksStale() && Date.now() - lastHoverScan > 3000) { lastHoverScan = Date.now(); scan(); }
      let el = t;
      while (el && el !== document.body && !blocks.some((b) => b.el === el)) el = el.parentElement;
      if (el && el !== document.body) { hoverEl = el; hoverEl.classList.add("kokoro-alt-hover"); }
    }
  }, { passive: true });

  // ---------- panel UI (shadow DOM so the site's CSS can't break it) ----------
  const host = document.createElement("div");
  host.id = "kokoro-reader-host";
  host.style.cssText = "position:fixed;right:16px;bottom:16px;z-index:2147483647;";
  const shadow = host.attachShadow({ mode: "open" });
  shadow.innerHTML = `
    <style>
      :host { all: initial; }
      .panel { font: 13px -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; color: #f2f2f2;
        background: rgba(28,28,32,0.94); border-radius: 12px; box-shadow: 0 6px 24px rgba(0,0,0,.35);
        padding: 10px 12px; width: 270px; user-select: none; }
      .head { display:flex; align-items:center; justify-content:space-between; cursor: move; margin-bottom: 8px; }
      .title { font-weight: 600; letter-spacing: .2px; }
      .row { display:flex; align-items:center; gap:6px; margin: 6px 0; }
      button { background:#3a3a42; color:#fff; border:0; border-radius:8px; padding:6px 0; font-size:15px;
        cursor:pointer; flex:1; line-height:1; }
      button:hover { background:#4a4a55; }
      button.primary { background:#d4441c; }
      button.primary:hover { background:#e85a30; }
      button.mini { flex:0 0 auto; padding:3px 7px; font-size:12px; background:transparent; }
      select { flex:1; background:#2b2b31; color:#fff; border:1px solid #444; border-radius:6px; padding:4px; font-size:12px; }
      input[type=range] { flex:1; }
      label { font-size:12px; color:#ccc; }
      .status { font-size:11.5px; color:#bbb; min-height: 15px; margin-top: 4px; line-height:1.35; }
      .hint { font-size:10.5px; color:#888; margin-top:4px; }
      .collapsed .body { display:none; }
      .collapsed { width:auto; }
    </style>
    <div class="panel">
      <div class="head" id="drag"><span class="title">🔊 Kokoro Reader</span>
        <span><button class="mini" id="collapse" title="Collapse">–</button><button class="mini" id="close" title="Hide panel (click the toolbar icon to show it again)">✕</button></span></div>
      <div class="row">
        <button id="prev" title="Previous paragraph">⏮</button>
        <button id="play" class="primary" title="Play / Pause (Alt+Shift+P). Reads the selected text if there is a selection.">▶</button>
        <button id="stop" title="Stop">⏹</button>
        <button id="next" title="Next paragraph">⏭</button>
      </div>
      <div class="body">
        <div class="row"><label>Voice</label><select id="voice"></select></div>
        <div class="row"><label>Speed</label><input id="speed" type="range" min="0.8" max="1.5" step="0.05"><label id="speedv"></label></div>
        <div class="row"><label><input id="skipcode" type="checkbox"> Skip code blocks</label></div>
        <div class="status" id="status">Ready. Press Play, select text, or Option+click a paragraph.</div>
        <div class="hint">Local Kokoro TTS · 127.0.0.1:8880</div>
      </div>
    </div>`;
  const $ = (id) => shadow.getElementById(id);
  const panel = shadow.querySelector(".panel");

  function setStatus(t) { $("status").textContent = t; }
  function setState(s) {
    state = s;
    $("play").textContent = s === "playing" ? "⏸" : s === "loading" ? "…" : "▶";
  }

  function saveSettings() { chrome.storage.local.set({ kokoroSettings: settings }); }

  const isHidden = () => (IS_OREILLY ? !!settings.hidden : panelHidden);
  function setHidden(h) {
    if (IS_OREILLY) { settings.hidden = h; saveSettings(); } else panelHidden = h;
    applySettingsToUI();
  }
  function showPanel() { if (isHidden()) setHidden(false); }

  function applySettingsToUI() {
    $("speed").value = String(settings.speed);
    $("speedv").textContent = `${Number(settings.speed).toFixed(2)}×`;
    $("skipcode").checked = !!settings.skipCode;
    panel.classList.toggle("collapsed", !!settings.collapsed);
    $("collapse").textContent = settings.collapsed ? "+" : "–";
    host.style.display = isHidden() ? "none" : "";
    if ([...$("voice").options].some((o) => o.value === settings.voice)) $("voice").value = settings.voice;
  }

  function fillVoices(list) {
    const sel = $("voice");
    sel.innerHTML = "";
    for (const v of list) {
      const o = document.createElement("option");
      o.value = v.id; o.textContent = v.label || v.id;
      sel.appendChild(o);
    }
    if (!list.some((v) => v.id === settings.voice) && list.length) settings.voice = list[0].id;
    sel.value = settings.voice;
  }

  // Keep the page's text selection when pressing panel buttons.
  shadow.querySelectorAll("button").forEach((b) => b.addEventListener("mousedown", (e) => e.preventDefault()));
  $("play").addEventListener("click", () => togglePlay());
  $("stop").addEventListener("click", () => stop(true));
  $("prev").addEventListener("click", () => skip(-1));
  $("next").addEventListener("click", () => skip(+1));
  $("voice").addEventListener("change", (e) => { settings.voice = e.target.value; saveSettings(); clearCache(); if (state === "playing") playFrom(idx); });
  $("speed").addEventListener("input", (e) => { $("speedv").textContent = `${Number(e.target.value).toFixed(2)}×`; });
  $("speed").addEventListener("change", (e) => { settings.speed = Number(e.target.value); saveSettings(); clearCache(); if (idx + 1 < blocks.length && state === "playing") fetchAudio(idx + 1).catch(() => {}); });
  $("skipcode").addEventListener("change", (e) => {
    settings.skipCode = e.target.checked; saveSettings();
    if (mode === "selection") return;
    const cur = blocks[idx] && blocks[idx].el;
    scan();
    const j = blocks.findIndex((b) => b.el === cur);
    if (j >= 0) idx = j;
  });
  $("collapse").addEventListener("click", () => { settings.collapsed = !settings.collapsed; saveSettings(); applySettingsToUI(); });
  $("close").addEventListener("click", () => setHidden(true));

  // Dragging the panel by its header
  (() => {
    let sx, sy, sr, sb, dragging = false;
    $("drag").addEventListener("mousedown", (e) => {
      if (e.target.tagName === "BUTTON") return;
      dragging = true; sx = e.clientX; sy = e.clientY;
      sr = parseInt(host.style.right, 10); sb = parseInt(host.style.bottom, 10); e.preventDefault();
    });
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      host.style.right = `${Math.max(0, sr - (e.clientX - sx))}px`;
      host.style.bottom = `${Math.max(0, sb - (e.clientY - sy))}px`;
    });
    window.addEventListener("mouseup", () => { dragging = false; });
  })();

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (!msg) return;
    if (msg.type === "kokoro-ping") { sendResponse({ ok: true }); return; }
    if (msg.type === "toggle-play") { showPanel(); togglePlay(); }
    if (msg.type === "toggle-panel") setHidden(!isHidden());
    if (msg.type === "show-panel") showPanel();
    if (msg.type === "read-selection") {
      const range = msg.fromFrame ? null : currentSelection();
      readSelection(range, msg.text);
    }
  });

  // Single-page apps (O'Reilly and others): when the URL changes, stop and rescan.
  let lastUrl = location.href;
  setInterval(() => {
    if (location.href !== lastUrl) {
      lastUrl = location.href;
      stop(true); blocks = []; idx = 0; clearCache();
      setStatus("New page. Press Play to read from the first visible paragraph.");
    }
  }, 1000);

  // Debug/test hook (isolated world only; the page can't see it).
  window.__kokoroDebug = { scan: () => (scan(), { detection: lastDetection, blocks: blocks.map((b) => b.text) }), state: () => ({ state, idx, mode, n: blocks.length }) };

  // ---------- init ----------
  document.documentElement.appendChild(host);
  host.style.display = isHidden() ? "none" : "";
  chrome.storage.local.get("kokoroSettings", async (res) => {
    settings = { ...DEFAULTS, ...(res && res.kokoroSettings) };
    fillVoices(FALLBACK_VOICES);
    applySettingsToUI();
    const r = await send({ type: "voices" });
    if (r.ok && r.data && r.data.voices && r.data.voices.length) {
      fillVoices(r.data.voices); applySettingsToUI();
    } else {
      setStatus("TTS server not reachable. Run ~/kokoro-tts/start.sh, then press Play again.");
    }
  });
})();
