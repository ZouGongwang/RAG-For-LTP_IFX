/* Corner chat for the local page and for a Confluence footer script.
   The API base is data-api, or the host this file was loaded from. */
(function () {
  const script = document.currentScript;
  const api = ((script && script.dataset.api) || (script ? new URL(script.src).origin : "")).replace(/\/$/, "");
  const space = (script && script.dataset.space) || "";

  function mount() {
    if (document.getElementById("bsp-assistant")) return;
    const host = document.createElement("div");
    host.id = "bsp-assistant";
    const root = host.attachShadow({ mode: "open" });
    root.innerHTML = `
<style>
  :host { all: initial; }
  :host, button, textarea { font: 15px/1.45 "Segoe UI", system-ui, sans-serif; }
  #bsp-launcher, #bsp-panel { --ink: #14222b; --muted: #5c6b76; --line: #d7dee4; --panel: #fff; --wash: #f3f6f8; --accent: #0c4a5e; --accent-2: #14627a; }
  #bsp-launcher {
    position: fixed; right: 1.25rem; bottom: 1.25rem; z-index: 2147483000;
    width: 3.5rem; height: 3.5rem; border: 0; border-radius: 999px;
    background: var(--accent); color: white; cursor: pointer;
    box-shadow: 0 8px 24px rgba(12, 74, 94, 0.28);
  }
  #bsp-launcher:hover { background: var(--accent-2); }
  #bsp-launcher svg { width: 1.5rem; height: 1.5rem; display: block; margin: 0 auto; }
  #bsp-panel {
    position: fixed; right: 1.25rem; bottom: 5.4rem; z-index: 2147483000;
    width: min(42rem, calc(100vw - 2rem)); height: min(36rem, calc(100vh - 7.5rem));
    display: flex; flex-direction: column; background: var(--panel); color: var(--ink);
    border: 1px solid var(--line); border-radius: 14px; overflow: hidden;
    box-shadow: 0 16px 48px rgba(20, 34, 43, 0.18);
  }
  #bsp-panel[hidden] { display: none; }
  header { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; padding: 0.75rem 0.85rem; background: var(--accent); color: white; }
  header strong { display: block; font-size: 0.95rem; }
  header span { display: block; font-size: 0.75rem; opacity: 0.85; font-weight: 400; }
  #bsp-close { border: 0; background: transparent; color: white; font-size: 1.3rem; line-height: 1; cursor: pointer; padding: 0.2rem 0.35rem; }
  #bsp-log { flex: 1; overflow: auto; padding: 0.85rem; display: flex; flex-direction: column; gap: 0.65rem; background: var(--wash); }
  .msg { padding: 0.7rem 0.85rem; border-radius: 10px; }
  .msg.you { align-self: flex-end; max-width: 85%; background: var(--accent); color: white; white-space: pre-wrap; }
  .msg.bot { align-self: stretch; background: white; border: 1px solid var(--line); line-height: 1.5; }
  .msg.bot p { margin: 0.4rem 0; }
  .msg.bot p:first-child { margin-top: 0; }
  .msg.bot h3, .msg.bot h4, .msg.bot h5 { font-size: 0.95rem; margin: 0.7rem 0 0.25rem; }
  .msg.bot ul, .msg.bot ol { margin: 0.35rem 0; padding-left: 1.25rem; }
  .msg.bot li { margin: 0.15rem 0; }
  .msg.bot code { font-family: ui-monospace, "Cascadia Code", monospace; font-size: 0.85em; }
  .msg.bot pre { margin: 0.4rem 0; padding: 0.45rem 0.6rem; overflow: auto; background: var(--wash); border-radius: 6px; }
  .table-wrap { overflow-x: auto; margin: 0.4rem 0; }
  .msg.bot table { border-collapse: collapse; width: 100%; font-size: 0.82rem; }
  .msg.bot th, .msg.bot td { border: 1px solid var(--line); padding: 0.28rem 0.45rem; text-align: left; vertical-align: top; }
  .msg.bot th { background: #e7eef1; }
  .msg.pending { color: var(--muted); align-self: stretch; }
  .sources { margin: 0.7rem 0 0; padding: 0.55rem 0 0; list-style: none; font-size: 0.8rem; border-top: 1px solid var(--line); }
  .sources::before { content: "Sources"; display: block; font-weight: 650; margin-bottom: 0.25rem; color: var(--ink); }
  .sources a { color: var(--accent-2); }
  form { display: flex; gap: 0.45rem; padding: 0.7rem; border-top: 1px solid var(--line); background: white; }
  textarea { flex: 1; resize: none; height: 2.7rem; border: 1px solid var(--line); border-radius: 8px; padding: 0.5rem 0.6rem; }
  form button { border: 0; border-radius: 8px; background: var(--accent); color: white; padding: 0 0.9rem; cursor: pointer; }
  form button:disabled { opacity: 0.55; cursor: default; }
</style>
<button id="bsp-launcher" type="button" aria-expanded="false" aria-controls="bsp-panel" aria-label="Open assistant">
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">
    <path d="M5 6.5A3.5 3.5 0 0 1 8.5 3h7A3.5 3.5 0 0 1 19 6.5v6A3.5 3.5 0 0 1 15.5 16H9l-4 3.5V6.5z"/>
  </svg>
</button>
<section id="bsp-panel" hidden>
  <header>
    <div>
      <strong>BSP assistant</strong>
      <span>Here to help you with your questions</span>
    </div>
    <button id="bsp-close" type="button" aria-label="Close">×</button>
  </header>
  <div id="bsp-log"></div>
  <form id="bsp-form">
    <textarea id="bsp-input" placeholder="Ask a question…" required></textarea>
    <button type="submit">Send</button>
  </form>
</section>`;

    const launcher = root.getElementById("bsp-launcher");
    const panel = root.getElementById("bsp-panel");
    const log = root.getElementById("bsp-log");
    const form = root.getElementById("bsp-form");
    const input = root.getElementById("bsp-input");

    function setOpen(open) {
      panel.hidden = !open;
      launcher.setAttribute("aria-expanded", String(open));
      launcher.setAttribute("aria-label", open ? "Close assistant" : "Open assistant");
      if (open) input.focus();
    }

    launcher.addEventListener("click", () => setOpen(panel.hidden));
    root.getElementById("bsp-close").addEventListener("click", () => setOpen(false));
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !panel.hidden) setOpen(false);
    });

    function addMessage(role, text) {
      const el = document.createElement("div");
      el.className = "msg " + role;
      el.textContent = text;
      log.appendChild(el);
      log.scrollTop = log.scrollHeight;
      return el;
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[ch]));
    }

    function inline(value) {
      return escapeHtml(value)
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
        .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    }

    function tableHtml(rows) {
      const cells = rows.map((row) => row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim()));
      const body = cells.filter((row) => !row.every((cell) => /^:?-+:?$/.test(cell)));
      if (!body.length) return "";
      const head = body[0].map((cell) => "<th>" + inline(cell) + "</th>").join("");
      const rest = body.slice(1).map((row) => "<tr>" + row.map((cell) => "<td>" + inline(cell) + "</td>").join("") + "</tr>").join("");
      return '<div class="table-wrap"><table><thead><tr>' + head + "</tr></thead><tbody>" + rest + "</tbody></table></div>";
    }

    function markdownToHtml(text) {
      const lines = String(text || "").replace(/\r\n/g, "\n").split("\n");
      let html = "";
      let i = 0;
      const paragraph = [];
      const flush = () => {
        if (!paragraph.length) return;
        html += "<p>" + inline(paragraph.join(" ")) + "</p>";
        paragraph.length = 0;
      };
      while (i < lines.length) {
        const line = lines[i];
        if (line.trim().startsWith("```")) {
          flush();
          const code = [];
          i += 1;
          while (i < lines.length && !lines[i].trim().startsWith("```")) code.push(escapeHtml(lines[i++]));
          html += "<pre><code>" + code.join("\n") + "</code></pre>";
          i += 1;
          continue;
        }
        if (/^\s*\|.+\|\s*$/.test(line)) {
          flush();
          const rows = [];
          while (i < lines.length && /^\s*\|.+\|\s*$/.test(lines[i])) rows.push(lines[i++]);
          html += tableHtml(rows);
          continue;
        }
        const heading = /^(#{1,3})\s+(.*)$/.exec(line);
        if (heading) {
          flush();
          const level = Math.min(heading[1].length + 2, 5);
          html += "<h" + level + ">" + inline(heading[2]) + "</h" + level + ">";
          i += 1;
          continue;
        }
        if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
          flush();
          const ordered = /^\s*\d+\./.test(line);
          const pattern = ordered ? /^\s*\d+\.\s+(.*)$/ : /^\s*[-*]\s+(.*)$/;
          const items = [];
          while (i < lines.length && pattern.test(lines[i])) items.push(inline(lines[i++].match(pattern)[1]));
          const tag = ordered ? "ol" : "ul";
          html += "<" + tag + ">" + items.map((item) => "<li>" + item + "</li>").join("") + "</" + tag + ">";
          continue;
        }
        if (!line.trim()) { flush(); i += 1; continue; }
        paragraph.push(line.trim());
        i += 1;
      }
      flush();
      return html || "<p>(empty answer)</p>";
    }

    function sourceList(sources) {
      const list = document.createElement("ul");
      list.className = "sources";
      for (const source of sources || []) {
        const item = document.createElement("li");
        const label = source.label || source.title || "Source";
        if (source.url) {
          const link = document.createElement("a");
          link.href = source.url;
          link.target = "_blank";
          link.rel = "noopener";
          link.textContent = label;
          item.appendChild(link);
        } else {
          item.textContent = label;
        }
        list.appendChild(item);
      }
      return list;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const question = input.value.trim();
      if (!question) return;
      input.value = "";
      addMessage("you", question);
      const pending = addMessage("bot pending", "Looking through the indexed pages…");
      const send = form.querySelector("button");
      send.disabled = true;
      let answer = "";
      let body = null;
      try {
        const payload = { question };
        if (space) payload.space = space;
        const resp = await fetch(api + "/query/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (!resp.ok || !resp.body) {
          const data = await resp.json().catch(() => ({}));
          const detail = data.detail;
          throw new Error(typeof detail === "string" ? detail : resp.statusText);
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
          const chunk = await reader.read();
          if (chunk.done) break;
          buffer += decoder.decode(chunk.value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop();
          for (const line of lines) {
            if (!line.trim()) continue;
            const msg = JSON.parse(line);
            if (msg.type === "error") throw new Error(msg.detail || "request failed");
            if (msg.type === "delta") {
              answer += msg.text;
              if (!body) {
                pending.classList.remove("pending");
                pending.textContent = "";
                body = document.createElement("div");
                pending.appendChild(body);
              }
              body.innerHTML = markdownToHtml(answer);
              log.scrollTop = log.scrollHeight;
            }
            if (msg.type === "done" && (msg.sources || []).length) pending.appendChild(sourceList(msg.sources));
          }
        }
        if (!answer) throw new Error("empty answer");
      } catch (err) {
        pending.classList.remove("pending");
        pending.textContent = "Could not get an answer. " + err.message;
      } finally {
        send.disabled = false;
        input.focus();
        log.scrollTop = log.scrollHeight;
      }
    });

    document.body.appendChild(host);
  }

  if (document.body) mount();
  else document.addEventListener("DOMContentLoaded", mount);
})();
