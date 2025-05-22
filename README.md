# AI Chatbot with Groq, LangGraph, and Tools

A streaming AI chatbot built using:
- [Groq API](https://console.groq.com/) to run blazing-fast open-source LLMs (like Gemma, LLaMA, etc.)
- [LangGraph](https://docs.langgraph.dev/) for stateful conversational workflows and tool usage
- [FastAPI](https://fastapi.tiangolo.com/) for the backend server with Server-Sent Events (SSE)
- [React.js](https://react.dev/) frontend with auto-scrolling chat interface
- Deployed to [Vercel](https://vercel.com/) (both frontend and backend)

---

## 🧠 Features

- ✅ LLM-powered assistant using Groq's ultra-fast API
- 🛠️ Supports external tools (Wikipedia + Arxiv search)
- 🔄 Real-time streaming response via SSE
- 🌐 Deployed with Vercel (frontend + backend)
- 💬 Simple and clean React-based chat interface

---

## 🚀 Live Demo

- **Frontend**: [https://ai-chat-bot-e7rc.vercel.app](https://ai-chat-bot-e7rc.vercel.app)
- **Backend**: [https://ai-chat-bot-ten-delta.vercel.app](https://ai-chat-bot-ten-delta.vercel.app)

---

## 🏗️ Tech Stack

- **LLM**: Groq API (e.g., `Gemma2-9b-It`)
- **Orchestration**: LangGraph (`StateGraph`, `ToolNode`)
- **Tools**: Langchain wrappers for Wikipedia and Arxiv
- **Backend**: FastAPI + StreamingResponse (SSE)
- **Frontend**: React.js + `fetch()` streaming
- **Deployment**: Vercel

---

## 🛠️ Setup Instructions

### Backend

```bash
git clone https://github.com/your-username/your-repo-name
cd backend-folder
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --reload
