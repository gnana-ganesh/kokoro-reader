// Kokoro Reader service worker.
// - All calls to the local TTS server go through here (host_permissions lets it reach
//   http://127.0.0.1:8880 without CORS / mixed-content issues; the server only accepts
//   chrome-extension:// origins).
// - On learning.oreilly.com the content script is always present (manifest).
// - On any other page nothing runs until you ask: toolbar icon, Alt+Shift+P, or the
//   "Read aloud with Kokoro" context menu inject the reader via activeTab + scripting.
const SERVER = "http://127.0.0.1:8880";
const MENU_ID = "kokoro-read-selection";

function toBase64(buf) {
  const bytes = new Uint8Array(buf);
  let bin = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(bin);
}

async function handle(msg) {
  if (msg.type === "health") {
    const r = await fetch(`${SERVER}/health`);
    return { ok: r.ok, data: await r.json() };
  }
  if (msg.type === "voices") {
    const r = await fetch(`${SERVER}/voices`);
    return { ok: r.ok, data: await r.json() };
  }
  if (msg.type === "tts") {
    const r = await fetch(`${SERVER}/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: msg.text, voice: msg.voice, speed: msg.speed }),
    });
    if (!r.ok) return { ok: false, error: `HTTP ${r.status}: ${await r.text()}` };
    return { ok: true, b64: toBase64(await r.arrayBuffer()) };
  }
  return null; // not a server request
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || !["health", "voices", "tts"].includes(msg.type)) return false;
  handle(msg)
    .then(sendResponse)
    .catch((e) => sendResponse({ ok: false, error: String(e && e.message ? e.message : e), offline: true }));
  return true; // async response
});

// ---------- on-demand injection ----------
function isInjectable(url) {
  return /^(https?|file):/i.test(url || "") && !/^https:\/\/chromewebstore\.google\.com\//i.test(url || "");
}

async function ping(tabId, frameId = 0) {
  try {
    const r = await chrome.tabs.sendMessage(tabId, { type: "kokoro-ping" }, { frameId });
    return !!(r && r.ok);
  } catch (_) {
    return false;
  }
}

// Make sure the reader is loaded in the tab (frame). Returns true if it is ready.
async function ensureInjected(tab, frameId = 0) {
  if (!tab || tab.id == null) return false;
  if (await ping(tab.id, frameId)) return true;
  if (tab.url && !isInjectable(tab.url)) return false;
  try {
    const target = { tabId: tab.id, frameIds: [frameId] };
    await chrome.scripting.insertCSS({ target, files: ["content.css"] });
    await chrome.scripting.executeScript({ target, files: ["Readability.js", "content.js"] });
    return await ping(tab.id, frameId);
  } catch (e) {
    console.warn("Kokoro Reader: cannot run on this page:", e && e.message);
    return false;
  }
}

async function sendTo(tab, message, frameId = 0) {
  try { return await chrome.tabs.sendMessage(tab.id, message, { frameId }); } catch (_) { return null; }
}

function flagUnsupported(tab) {
  if (!tab || tab.id == null) return;
  chrome.action.setBadgeBackgroundColor({ color: "#888", tabId: tab.id }).catch(() => {});
  chrome.action.setBadgeText({ text: "✕", tabId: tab.id }).catch(() => {});
  chrome.action.setTitle({ title: "Kokoro Reader can't run on this page (browser/internal page)", tabId: tab.id }).catch(() => {});
}

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

// Toolbar icon: inject + show the panel; if already there, show/hide it.
chrome.action.onClicked.addListener(async (tab) => {
  const already = await ping(tab.id);
  if (!already) {
    if (await ensureInjected(tab)) await sendTo(tab, { type: "show-panel" });
    else flagUnsupported(tab);
    return;
  }
  await sendTo(tab, { type: "toggle-panel" });
});

// Alt+Shift+P: inject if needed, then play / pause.
chrome.commands.onCommand.addListener(async (command, tabArg) => {
  if (command !== "toggle-play") return;
  const tab = tabArg || (await activeTab());
  if (await ensureInjected(tab)) await sendTo(tab, { type: "toggle-play" });
  else flagUnsupported(tab);
});

// Right-click "Read aloud with Kokoro" on selected text.
function createMenu() {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({ id: MENU_ID, title: "Read aloud with Kokoro", contexts: ["selection"] });
  });
}
chrome.runtime.onInstalled.addListener(createMenu);
chrome.runtime.onStartup.addListener(createMenu);

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== MENU_ID || !tab) return;
  const text = info.selectionText || "";
  const frameId = info.frameId || 0;
  // Prefer the frame that holds the selection (so it can be highlighted); fall back to the top frame.
  if (frameId && (await ensureInjected(tab, frameId))) {
    await sendTo(tab, { type: "read-selection", text }, frameId);
    return;
  }
  if (await ensureInjected(tab, 0)) await sendTo(tab, { type: "read-selection", text, fromFrame: frameId }, 0);
  else flagUnsupported(tab);
});
