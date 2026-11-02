// Phone chat for the local assistant. Plain JS, no libraries, no external requests.
// Server text is always inserted with textContent, never as HTML.
"use strict";

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const MAX_DIRECT_UPLOAD = 3.5 * 1024 * 1024;   // smaller photos go up as they are; bigger ones are scaled down
const MAX_SIDE = 2000;
let photo = null;          // {dataUrl, previewUrl, name}
let busy = false;

async function api(path, body) {
  const opts = { method: body ? "POST" : "GET", credentials: "same-origin", headers: {} };
  if (body) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch (e) {
    return { status: 0, data: { error: "Can't reach the laptop. Is the server running, and are you on the same network?" } };
  }
  let data = {};
  try { data = await res.json(); } catch (e) { /* empty body */ }
  return { status: res.status, data };
}

function show(view) {
  $("#login").hidden = view !== "login";
  $("#chat").hidden = view !== "chat";
  $("#composer").hidden = view !== "chat";
  $("#signout").hidden = view !== "chat";
}

function scrollDown() {
  requestAnimationFrame(() => window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" }));
}

function chips(list, onPick) {
  const box = el("div", "chips");
  for (const text of list || []) {
    const b = el("button", "chip", text);
    b.type = "button";
    b.addEventListener("click", () => onPick(text));
    box.appendChild(b);
  }
  return box;
}

function proposalBox(p) {
  const box = el("div", "proposal");
  box.dataset.pid = p.id;
  box.appendChild(el("div", "label", "Proposed calendar add"));
  box.appendChild(el("p", "what", p.summary));
  box.appendChild(el("p", "when", p.when));
  setProposalStatus(box, p.status || "open");
  return box;
}

function setProposalStatus(box, status) {
  box.querySelector(".actions")?.remove();
  box.querySelector(".status")?.remove();
  if (status === "open") {
    const row = el("div", "actions");
    const ok = el("button", "primary", "Confirm");
    const no = el("button", "ghost", "Cancel");
    ok.type = no.type = "button";
    ok.addEventListener("click", () => decide(box, "confirm"));
    no.addEventListener("click", () => decide(box, "cancel"));
    row.append(ok, no);
    box.appendChild(row);
  } else {
    const words = { confirmed: "Added to your calendar", cancelled: "Cancelled: nothing added",
                    rejected: "No longer open: nothing changed", error: "Not added (calendar error)" };
    box.appendChild(el("div", "status " + status, words[status] || status));
  }
}

function addUser(text, hasPhoto, previewUrl) {
  const li = el("li", "msg user");
  if (previewUrl) {
    const img = el("img", "thumb");
    img.src = previewUrl;
    img.alt = "Photo you sent";
    li.appendChild(img);
  } else if (hasPhoto) {
    li.appendChild(el("span", "pill", "Photo"));
    li.appendChild(document.createElement("br"));
  }
  if (text) li.appendChild(document.createTextNode(text));
  $("#log").appendChild(li);
  $("#intro").hidden = true;
  scrollDown();
}

function addBot(m) {
  const li = el("li", "msg bot" + (m.outcome ? " o-" + m.outcome : ""));
  li.appendChild(document.createTextNode(m.text || ""));
  if (m.cards && m.cards.length) {
    const box = el("div", "cards");
    m.cards.forEach((c, i) => {
      const d = el("details");
      d.appendChild(el("summary", "", `${i + 1}. ${c.q}`));
      d.appendChild(el("p", "", c.a));
      box.appendChild(d);
    });
    li.appendChild(box);
  }
  if (m.proposal) li.appendChild(proposalBox(m.proposal));
  if (m.suggestions && m.suggestions.length) li.appendChild(chips(m.suggestions, (t) => send(t)));
  const bits = [];
  if (m.intent) bits.push(m.intent.replace("_", " "));
  if (typeof m.latency_s === "number") bits.push(`${m.latency_s.toFixed(1)} s on the laptop`);
  if (bits.length) li.appendChild(el("span", "meta", bits.join(" · ")));
  $("#log").appendChild(li);
  $("#intro").hidden = true;
  scrollDown();
}

function typing(on) {
  $("#typing")?.remove();
  if (!on) return;
  const li = el("li", "msg bot typing");
  li.id = "typing";
  li.append(el("i"), el("i"), el("i"));
  $("#log").appendChild(li);
  scrollDown();
}

async function boot() {
  const { status, data } = await api("/api/state");
  if (status === 401) return show("login");
  if (status !== 200) {
    show("login");
    $("#login-error").textContent = data.error || "Can't reach the laptop.";
    return;
  }
  show("chat");
  $("#clock").textContent = data.clock || "Runs on your laptop";
  const caps = $("#capabilities");
  caps.replaceChildren(chips(data.capabilities.map((c) => c.split(" (")[0].split(" / ")[0]), (t) => pick(t)));
  $("#log").replaceChildren();
  for (const m of data.history || []) (m.role === "user" ? addUser(m.text, m.photo) : addBot(m));
  const samples = data.samples || [];
  $("#sample-chips").replaceChildren(chips(samples, (name) => useSample(name)));
  $("#samples").dataset.available = samples.length ? "1" : "";
}

function pick(text) {
  // A capability chip: complete requests go straight out; the two that need more from you fill the box.
  if (/^Add to my calendar/.test(text)) {
    $("#text").value = "Add to my calendar: ";
    return $("#text").focus();
  }
  if (/^Split/.test(text)) {
    $("#text").value = "Split this receipt";
    return document.querySelector('label[for="photo"]').click();
  }
  send(text);
}

async function send(text) {
  if (busy) return;
  text = (text ?? $("#text").value).trim();
  if (!text && !photo) return;
  busy = true;
  $("#send").disabled = true;
  const sending = photo;
  addUser(text, !!sending, sending && sending.previewUrl);
  $("#text").value = "";
  clearPhoto(false);
  typing(true);
  const { status, data } = await api("/api/message", { text, photo: sending ? sending.dataUrl : null });
  typing(false);
  busy = false;
  $("#send").disabled = false;
  if (status === 401) return show("login");
  if (status !== 200) return addBot({ text: data.error || `Error ${status}`, outcome: "error" });
  addBot(data);
}

async function decide(box, decision) {
  box.querySelectorAll("button").forEach((b) => (b.disabled = true));
  const { status, data } = await api("/api/decide", { proposal_id: box.dataset.pid, decision });
  if (status === 401) return show("login");
  if (status !== 200) {
    box.querySelectorAll("button").forEach((b) => (b.disabled = false));
    return addBot({ text: data.error || `Error ${status}`, outcome: "error" });
  }
  setProposalStatus(box, data.outcome);
  addBot(data);
}

function readAsDataURL(blob) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(blob);
  });
}

async function preparePhoto(file) {
  const direct = file.size <= MAX_DIRECT_UPLOAD && /^image\/(jpeg|png|webp)$/.test(file.type);
  if (direct) return { dataUrl: await readAsDataURL(file), previewUrl: URL.createObjectURL(file), name: file.name };
  const bitmap = await createImageBitmap(file);      // decodes HEIC on iPhone Safari too
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.9));
  return { dataUrl: await readAsDataURL(blob), previewUrl: URL.createObjectURL(blob), name: file.name };
}

async function setPhoto(file) {
  try {
    photo = await preparePhoto(file);
  } catch (e) {
    photo = null;
    return addBot({ text: "That photo couldn't be opened. Try a JPEG or PNG.", outcome: "error" });
  }
  $("#attach-img").src = photo.previewUrl;
  $("#attach-name").textContent = photo.name || "photo";
  $("#attach-preview").hidden = false;
  $("#samples").hidden = true;
  if (!$("#text").value) $("#text").value = "Split this receipt";
}

function clearPhoto(resetText = true) {
  photo = null;
  $("#photo").value = "";
  $("#attach-preview").hidden = true;
  if (resetText && $("#text").value === "Split this receipt") $("#text").value = "";
}

async function useSample(name) {
  const res = await fetch(`/api/sample/${encodeURIComponent(name)}`, { credentials: "same-origin" });
  if (!res.ok) return addBot({ text: "Couldn't load that sample.", outcome: "error" });
  const blob = await res.blob();
  // Same path as a photo from the camera roll: it is uploaded with the message, not read on the server.
  await setPhoto(new File([blob], `${name}.png`, { type: "image/png" }));
}

document.addEventListener("DOMContentLoaded", () => {
  $("#login-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    $("#login-error").textContent = "";
    const { status, data } = await api("/api/login", { token: $("#token").value });
    if (status === 200) {
      $("#token").value = "";
      return boot();
    }
    $("#login-error").textContent = data.error || `Error ${status}`;
  });
  $("#send-form").addEventListener("submit", (e) => { e.preventDefault(); send(); });
  $("#photo").addEventListener("change", (e) => { if (e.target.files[0]) setPhoto(e.target.files[0]); });
  $("#attach-clear").addEventListener("click", () => clearPhoto());
  document.querySelector('label[for="photo"]').addEventListener("click", (e) => {
    // With sample receipts on (demo mode), the first tap shows them; a second tap opens the picker.
    if ($("#samples").dataset.available && $("#samples").hidden && !photo) {
      e.preventDefault();
      $("#samples").hidden = false;
    }
  });
  $("#signout").addEventListener("click", async () => {
    await api("/api/logout", {});
    show("login");
  });
  boot();
});
