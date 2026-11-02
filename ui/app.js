const params = new URLSearchParams(location.search);
const orchestratorBase = (
  params.get("orchestrator") ||
  localStorage.getItem("orchestrator_url") ||
  ""
).replace(/\/$/, "");

function addLine(role, text) {
  const log = document.getElementById("log");
  const item = document.createElement("li");
  const tag = document.createElement("div");
  tag.className = "role";
  tag.textContent = role;
  const body = document.createElement("div");
  body.textContent = text;
  item.append(tag, body);
  log.append(item);
  log.scrollTop = log.scrollHeight;
}

async function loadConfig() {
  if (!orchestratorBase) {
    addLine("system", "Set ?orchestrator=http://127.0.0.1:8001 or localStorage.orchestrator_url");
    return {};
  }
  const res = await fetch(`${orchestratorBase}/ui-config`);
  if (!res.ok) throw new Error(`ui-config ${res.status}`);
  return res.json();
}

function parseSseChunk(raw) {
  const lines = String(raw || "").split("\n");
  const dataLines = lines.filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trim());
  if (!dataLines.length) return null;
  try {
    return JSON.parse(dataLines.join("\n"));
  } catch {
    return { text: dataLines.join("\n") };
  }
}

async function streamChat(cfg, sessionId, userId, message) {
  const res = await fetch(`${orchestratorBase}/chat/stream`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      user_id: userId,
      message,
      orchestrator_id: cfg.default_orchestrator || "",
      model: cfg.chat_logical_id || undefined,
    }),
  });
  if (!res.ok) throw new Error(`chat/stream ${res.status}`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const event = parseSseChunk(part);
      if (!event) continue;
      const text = event.answer || event.text || event.content || "";
      if (text) {
        answer += typeof text === "string" ? text : JSON.stringify(text);
      }
    }
  }
  return answer;
}

async function main() {
  const cfg = await loadConfig();
  const title = document.getElementById("app-title");
  const meta = document.getElementById("engine-meta");
  title.textContent = cfg.title || "RLM Chat";
  meta.textContent = [
    cfg.default_orchestrator && `engine=${cfg.default_orchestrator}`,
    cfg.chat_logical_id && `chat=${cfg.chat_logical_id}`,
  ].filter(Boolean).join(" · ");

  const userId = localStorage.getItem("user_id") || `user_${crypto.randomUUID()}`;
  localStorage.setItem("user_id", userId);
  let sessionId = localStorage.getItem("session_id") || crypto.randomUUID();
  localStorage.setItem("session_id", sessionId);

  const storageUrl = (cfg.storage_url || "").replace(/\/$/, "");
  if (storageUrl) {
    try {
      const created = await fetch(`${storageUrl}/chat/sessions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          title: "RLM chat",
          model: cfg.chat_logical_id || "",
        }),
      });
      if (created.ok) {
        const body = await created.json();
        if (body.id || body.session_id) {
          sessionId = body.id || body.session_id;
          localStorage.setItem("session_id", sessionId);
        }
      }
    } catch {
      // Storage may be down during unit-test-only work; stream still uses local session id.
    }
  }

  document.getElementById("composer").addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = document.getElementById("message");
    const message = input.value.trim();
    if (!message) return;
    input.value = "";
    addLine("you", message);
    try {
      const answer = await streamChat(cfg, sessionId, userId, message);
      addLine("agent", answer || "(empty)");
    } catch (err) {
      addLine("error", String(err));
    }
  });
}

main().catch((err) => addLine("error", String(err)));
