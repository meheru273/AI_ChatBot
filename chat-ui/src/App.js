import React, { useState, useEffect, useRef, useCallback } from "react";
import "./App.css";
import { streamChat, checkHealth } from "./api/chatStream";

/** Friendly labels for the tool names the backend streams. */
const TOOL_LABELS = {
  search_menu: "Checking the menu",
  filter_menu: "Filtering by your requirements",
  list_menu_categories: "Looking through the categories",
  get_restaurant_info: "Looking up restaurant details",
  search_knowledge: "Searching our knowledge base",
  wikipedia: "Searching Wikipedia",
  arxiv: "Searching arXiv",
};

const toolLabel = (name) => TOOL_LABELS[name] || `Running ${name}`;

const now = () =>
  new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [activeTool, setActiveTool] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState("checking");
  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const runHealthCheck = useCallback(async () => {
    setConnectionStatus("checking");
    try {
      const health = await checkHealth();
      setConnectionStatus(health.status === "healthy" ? "connected" : "degraded");
    } catch (error) {
      console.error("Health check failed:", error);
      setConnectionStatus("error");
    }
  }, []);

  useEffect(() => {
    runHealthCheck();
  }, [runHealthCheck]);

  const sendMessage = async () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    abortControllerRef.current = new AbortController();

    const userMessage = { role: "user", content: trimmed, time: now() };
    const history = [...messages, userMessage];

    // The placeholder is appended immediately and filled in as tokens land,
    // so the reply appears to type itself rather than arriving all at once.
    setMessages([...history, { role: "assistant", content: "", time: now(), streaming: true }]);
    setInput("");
    setIsLoading(true);
    setActiveTool(null);

    const updateAssistant = (updater) =>
      setMessages((prev) => {
        const next = [...prev];
        const last = next.length - 1;
        next[last] = { ...next[last], ...updater(next[last]) };
        return next;
      });

    try {
      const finalText = await streamChat({
        // Strip UI-only fields before sending.
        messages: history.map(({ role, content }) => ({ role, content })),
        signal: abortControllerRef.current.signal,
        onToken: (_chunk, assembled) => updateAssistant(() => ({ content: assembled })),
        onToolStart: (tool) => setActiveTool(tool),
        onToolEnd: () => setActiveTool(null),
      });

      updateAssistant(() => ({ content: finalText, streaming: false }));
      setConnectionStatus("connected");
    } catch (error) {
      if (error.name === "AbortError") {
        updateAssistant((current) => ({
          content: current.content || "_Cancelled._",
          streaming: false,
        }));
        return;
      }

      console.error("Error sending message:", error);
      const isNetworkError = error instanceof TypeError;
      if (isNetworkError) setConnectionStatus("error");

      updateAssistant((current) => ({
        streaming: false,
        content:
          current.content ||
          (isNetworkError
            ? "I can't reach the server. Check that the backend is running, then try again."
            : `Sorry — ${error.message}`),
      }));
    } finally {
      setIsLoading(false);
      setActiveTool(null);
      abortControllerRef.current = null;
    }
  };

  const cancelRequest = () => {
    abortControllerRef.current?.abort();
    setIsLoading(false);
    setActiveTool(null);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => setMessages([]);

  const statusMeta = {
    connected: { color: "#10b981", text: "Connected" },
    degraded: { color: "#f59e0b", text: "Degraded" },
    error: { color: "#ef4444", text: "Disconnected" },
    checking: { color: "#f59e0b", text: "Connecting..." },
  }[connectionStatus];

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="app-logo">🍽️</div>
        <h1 className="app-title">Banchan Assistant</h1>
        <div className="connection-status">
          <div
            className="status-indicator"
            style={{ backgroundColor: statusMeta.color }}
            title={statusMeta.text}
          />
          <span className="status-text">{statusMeta.text}</span>
          <button className="clear-chat-btn" onClick={clearChat} title="Clear chat">
            🗑️
          </button>
        </div>
      </header>

      <div className="messages-container">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Ask me about the menu, allergens, or delivery.</p>
            <ul>
              <li>"I'm allergic to sesame — what can I eat?"</li>
              <li>"What's the difference between japchae and gimbap?"</li>
              <li>"What do you have under £6?"</li>
            </ul>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`message ${
              msg.role === "user" ? "message-user" : "message-assistant"
            }`}
          >
            <div className="message-header">
              <span className="message-role">
                {msg.role === "user" ? "👤 You" : "🤖 Assistant"}
              </span>
              <span className="message-time">{msg.time}</span>
            </div>
            <div className="message-content">
              {msg.content.split("\n").map((line, idx) => (
                <React.Fragment key={idx}>
                  {line}
                  {idx < msg.content.split("\n").length - 1 && <br />}
                </React.Fragment>
              ))}
              {msg.streaming && <span className="streaming-cursor" />}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="loading-indicator">
            <div className="loading-spinner">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <circle
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="#4f46e5"
                  strokeWidth="4"
                  strokeDasharray="30 30"
                >
                  <animateTransform
                    attributeName="transform"
                    attributeType="XML"
                    type="rotate"
                    dur="1s"
                    from="0 12 12"
                    to="360 12 12"
                    repeatCount="indefinite"
                  />
                </circle>
              </svg>
            </div>
            <span>{activeTool ? `${toolLabel(activeTool)}…` : "Thinking…"}</span>
            <button className="cancel-btn" onClick={cancelRequest}>
              Cancel
            </button>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        <div className="input-wrapper">
          <textarea
            className="message-input"
            rows="2"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about the menu, allergens, or delivery…"
            disabled={isLoading}
            maxLength={1000}
          />
          <div className="input-footer">
            <span className="char-count">{input.length}/1000</span>
            <div className="button-group">
              <button
                className="retry-btn"
                onClick={runHealthCheck}
                disabled={isLoading}
                title="Check server connection"
              >
                🔄
              </button>
              <button
                className="send-button"
                onClick={sendMessage}
                disabled={isLoading || !input.trim() || connectionStatus === "error"}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <path
                    d="M22 2L11 13"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path
                    d="M22 2L15 22L11 13L2 9L22 2Z"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                {isLoading ? "Streaming…" : "Send"}
              </button>
            </div>
          </div>
        </div>
      </div>

      {connectionStatus === "error" && (
        <div className="error-banner">
          ⚠️ Can't reach the backend. Check that the server is running and that
          REACT_APP_API_URL is set correctly.
        </div>
      )}
    </div>
  );
}

export default App;
