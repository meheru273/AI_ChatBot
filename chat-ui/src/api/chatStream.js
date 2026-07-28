/**
 * Client for the backend's Server-Sent Events chat protocol.
 *
 * EventSource is not usable here: it only issues GET requests, and the
 * conversation has to go up in a POST body. So this reads the response stream
 * directly and parses the SSE wire format.
 *
 * Events emitted by the server:
 *   token       {content}  incremental model output
 *   tool_start  {tool}     a tool began running
 *   tool_end    {tool}     a tool returned
 *   done        {content}  full answer, terminal
 *   error       {message}  failure after the stream opened, terminal
 */

const API_URL = (
  process.env.REACT_APP_API_URL || "http://localhost:8000"
).replace(/\/$/, "");

export const apiUrl = (path) => `${API_URL}${path}`;

export async function checkHealth(signal) {
  const response = await fetch(apiUrl("/health"), {
    method: "GET",
    headers: { "Content-Type": "application/json" },
    signal,
  });
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }
  return response.json();
}

/** Split one SSE frame into its event name and parsed payload. */
function parseFrame(frame) {
  let event = "message";
  const dataLines = [];

  for (const line of frame.split("\n")) {
    if (line.startsWith(":")) continue; // keep-alive comment
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).replace(/^ /, ""));
    }
  }

  if (dataLines.length === 0) return null;

  try {
    return { event, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return { event, data: { raw: dataLines.join("\n") } };
  }
}

/**
 * Stream one assistant turn.
 *
 * Resolves with the full text once the server sends `done`. Rejects if the
 * server sends `error`, or if the connection fails.
 */
export async function streamChat({
  messages,
  signal,
  onToken = () => {},
  onToolStart = () => {},
  onToolEnd = () => {},
}) {
  const response = await fetch(apiUrl("/chat/stream"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
    signal,
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) detail = body.detail;
    } catch {
      /* non-JSON error body; keep the status text */
    }
    throw new Error(detail);
  }

  if (!response.body) {
    throw new Error("Streaming is not supported by this browser.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let assembled = "";
  let finished = false;

  try {
    while (!finished) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // A blank line terminates a frame. The trailing element is retained
      // because it may be a partial frame still arriving.
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const rawFrame of frames) {
        const parsed = parseFrame(rawFrame);
        if (!parsed) continue;

        const { event, data } = parsed;
        if (event === "token") {
          assembled += data.content ?? "";
          onToken(data.content ?? "", assembled);
        } else if (event === "tool_start") {
          onToolStart(data.tool);
        } else if (event === "tool_end") {
          onToolEnd(data.tool);
        } else if (event === "done") {
          // `done` carries the authoritative text; prefer it over our
          // accumulation in case a frame was dropped.
          assembled = data.content ?? assembled;
          finished = true;
          break;
        } else if (event === "error") {
          throw new Error(data.message || "The assistant failed mid-response.");
        }
      }
    }
  } finally {
    reader.cancel().catch(() => {});
  }

  if (!finished && !assembled) {
    throw new Error("The connection closed before a response arrived.");
  }

  return assembled;
}

/** Non-streaming fallback, same contract minus the incremental callbacks. */
export async function sendChat({ messages, signal }) {
  const response = await fetch(apiUrl("/chat"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Server error: ${response.status}`);
  }

  const data = await response.json();
  return data.response ?? "";
}
