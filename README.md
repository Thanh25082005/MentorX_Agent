# 🤖 MentorX Agent — AI Academy Chatbot

> Hệ thống AI Agent tư vấn học thuật thông minh, được xây dựng trên kiến trúc **LangGraph + ReAct Reasoning + RAG**, có khả năng tra cứu khóa học, tính toán học phí, tìm kiếm Internet và ghi nhớ ngữ cảnh hội thoại.

---

## 📑 Mục lục

- [Tổng quan](#-tổng-quan)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Pipeline xử lý](#-pipeline-xử-lý-chi-tiết)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cài đặt & Chạy](#-cài-đặt--chạy)
- [Biến môi trường](#-biến-môi-trường)
- [API Endpoints](#-api-endpoints)
- [Kịch bản Test](#-kịch-bản-test)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)

---

## 🎯 Tổng quan

MentorX Agent là một AI Chatbot chuyên tư vấn khóa học cho AI Academy, được thiết kế theo kiến trúc **Agentic AI** hiện đại với các khả năng:

| Tính năng | Mô tả |
|-----------|-------|
| 🧠 **Intent Classification** | Tự động phân loại câu hỏi vào 3 luồng xử lý |
| 🔧 **Tool Calling (ReAct)** | Gọi công cụ tự động: tìm khóa học, tính toán, tìm web |
| 📚 **RAG** | Truy vấn tài liệu nội bộ (syllabus, chính sách) bằng Vector Search |
| 🌐 **Web Search** | Tìm kiếm kiến thức từ Internet (DuckDuckGo) |
| 💬 **Conversation Memory** | Ghi nhớ ngữ cảnh hội thoại qua nhiều lượt |
| 📊 **Markdown Rendering** | Format bảng, list, heading, emoji trên giao diện |
| 🔍 **Planning Trace** | Hiển thị quá trình suy luận của Agent trên UI |

---

## 🏗 Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                    │
│              http://localhost:3000                       │
│  ┌─────────┐  ┌──────────┐  ┌────────────────────────┐  │
│  │ Chat UI │  │ Markdown │  │ Planning Trace Viewer  │  │
│  │         │  │ Renderer │  │ (Collapsible)          │  │
│  └────┬────┘  └──────────┘  └────────────────────────┘  │
│       │ SSE Stream (Server-Sent Events)                  │
└───────┼─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                 Backend (FastAPI)                        │
│              http://localhost:8000                       │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Orchestrator (LangGraph)             │   │
│  │                                                    │   │
│  │  ┌─────────────┐    ┌────────────────────────┐    │   │
│  │  │  Classify    │───▶│  Router (Conditional   │    │   │
│  │  │  Intent      │    │  Edges)                │    │   │
│  │  └─────────────┘    └───────┬────────────────┘    │   │
│  │                      ┌──────┼──────┐              │   │
│  │                      ▼      ▼      ▼              │   │
│  │               ┌──────┐ ┌────┐ ┌────────┐          │   │
│  │               │DIRECT│ │RAG │ │ ReAct  │          │   │
│  │               │Answer│ │    │ │ Loop   │          │   │
│  │               └──┬───┘ └─┬──┘ └───┬────┘          │   │
│  │                  │       │        │               │   │
│  │                  ▼       ▼        ▼               │   │
│  │            ┌──────────────────────────┐           │   │
│  │            │    Memory Update Node    │           │   │
│  │            └──────────────────────────┘           │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐   │
│  │ Qdrant   │  │  Groq    │  │   Tools              │   │
│  │ (Vector  │  │  LLM     │  │  ├─ course_search    │   │
│  │  Store)  │  │  API     │  │  ├─ calculator        │   │
│  └──────────┘  └──────────┘  │  └─ web_search        │   │
│                              └──────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## ⚙ Pipeline xử lý chi tiết

### Bước 1: User gửi tin nhắn
```
User → Frontend (Next.js) → POST /api/chat/stream → Backend (FastAPI)
```
Frontend gửi request SSE (Server-Sent Events) để nhận phản hồi dạng stream theo thời gian thực.

### Bước 2: Orchestrator nhận request
```python
orchestrator.handle_message(session_id, message, debug=True)
```
Orchestrator là trung tâm điều phối, được xây dựng bằng **LangGraph StateGraph** với các node:

### Bước 3: Classify Intent (Phân loại ý định)
LLM (Groq) phân tích câu hỏi và phân loại vào 1 trong 3 luồng:

| Intent | Khi nào | Ví dụ |
|--------|---------|-------|
| `DIRECT_ANSWER` | Chào hỏi, câu đơn giản | "Xin chào", "Cảm ơn" |
| `USE_RAG` | Hỏi về chính sách, lộ trình, tài liệu nội bộ | "Lộ trình học AI Academy?" |
| `USE_TOOLS` | Cần tra cứu, tính toán, tìm web | "Tìm khóa Python", "Giá Bitcoin?" |

### Bước 4a: Luồng DIRECT_ANSWER
```
classify_intent → direct → memory_update → END
```
- LLM trả lời trực tiếp không cần tra cứu
- Nhanh nhất (~1-2 giây)

### Bước 4b: Luồng USE_RAG
```
classify_intent → rag → memory_update → END
```
1. Tạo embedding từ câu hỏi (all-MiniLM-L6-v2)
2. Truy vấn Qdrant Vector Store tìm tài liệu tương đồng
3. Inject context vào prompt, LLM tổng hợp câu trả lời

### Bước 4c: Luồng USE_TOOLS (ReAct Loop)
```
classify_intent → react → memory_update → END
```
Agent chạy vòng lặp **ReAct** (Reasoning + Acting) tối đa 10 vòng:

```
┌─────────────────────────────────────────┐
│            ReAct Loop                   │
│                                         │
│  Thought → Action → Observation ──┐     │
│      ▲                            │     │
│      └────────────────────────────┘     │
│                                         │
│  Khi đủ thông tin → Final Answer        │
└─────────────────────────────────────────┘
```

**Các Tool có sẵn:**

| Tool | Chức năng | Input |
|------|-----------|-------|
| `course_search` | Tìm khóa học từ `courses.csv` | Keyword (tên, level, giảng viên) |
| `calculator` | Tính toán biểu thức toán học | Biểu thức (VD: `5000000 * 3`) |
| `web_search` | Tìm kiếm Internet (DuckDuckGo) | Query tìm kiếm |

### Bước 5: Memory Update
- LangGraph Checkpointer lưu trạng thái hội thoại theo `thread_id`
- Cho phép Agent nhớ ngữ cảnh qua nhiều lượt chat

### Bước 6: Stream Response
Backend gửi phản hồi về Frontend qua SSE với các event:

```
status: loading → thinking → streaming
metadata: {intent, tools_used, rag_used, trace}
token: {delta: "từng"} → {delta: "từ"} → {delta: "một"}
final: {answer: "câu trả lời hoàn chỉnh"}
done
```

### Bước 7: Frontend Render
- **ReactMarkdown** render Markdown (bảng, list, heading, code, link)
- **Planning Trace Viewer** hiển thị quá trình suy luận (collapsible)
- **Metadata Badges** hiển thị intent, tools, RAG status

---

## 📁 Cấu trúc thư mục

```
Demo_AIagent/
├── app/                          # Backend Python
│   ├── main.py                   # FastAPI entry point + lifespan
│   ├── api/
│   │   └── routes.py             # API endpoints (chat, stream, health)
│   ├── agent/
│   │   ├── orchestrator.py       # LangGraph Orchestrator (trung tâm điều phối)
│   │   ├── react_loop.py         # ReAct loop (raw Groq SDK)
│   │   └── react_loop_langchain.py # ReAct loop (LangChain structured output)
│   ├── core/
│   │   ├── config.py             # Pydantic Settings (đọc .env)
│   │   └── groq_client.py        # Groq API client wrapper
│   ├── memory/
│   │   └── short_term.py         # Session memory (InMemory / Redis backend)
│   ├── models/
│   │   └── schemas.py            # Pydantic models (Request/Response/Trace)
│   ├── rag/
│   │   └── retriever.py          # RAG retriever (Qdrant + Sentence Transformers)
│   └── tools/
│       ├── base.py               # BaseTool abstract class
│       ├── calculator.py         # Calculator tool
│       ├── course_search.py      # Course search tool (pandas + CSV)
│       └── web_search.py         # Web search tool (DuckDuckGo)
│
├── frontend/                     # Frontend Next.js
│   ├── app/                      # Next.js App Router
│   ├── components/
│   │   └── chat/
│   │       ├── chat-message.tsx  # Message bubble + Markdown + Planning Trace
│   │       └── ...               # Other chat components
│   ├── hooks/
│   │   └── use-agent-chat.ts     # Chat state management + SSE streaming
│   ├── lib/
│   │   └── api/
│   │       └── agent-client.ts   # Backend API client
│   └── types/
│       └── chat.ts               # TypeScript types
│
├── data/
│   ├── courses.csv               # Dữ liệu khóa học
│   └── docs/                     # Tài liệu RAG (syllabus.md, ...)
│
├── .env.example                  # Template biến môi trường
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── backend.Dockerfile            # Docker image cho Backend
├── frontend/frontend.Dockerfile  # Docker image cho Frontend
└── docker-compose.yml            # Docker Compose (Qdrant + Redis + Backend + Frontend)
```

---

## 🚀 Cài đặt & Chạy

### Yêu cầu hệ thống
- Python 3.12+
- Node.js 20+
- pnpm (hoặc npm/yarn)

### 1. Clone & cài đặt Backend

```bash
# Clone
git clone <repo-url>
cd Demo_AIagent

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Cấu hình biến môi trường

```bash
cp .env.example .env
# Mở .env và điền GROQ_API_KEY
```

### 3. Cài đặt Frontend

```bash
cd frontend
pnpm install    # hoặc: npm install
cd ..
```

### 4. Chạy project

**Terminal 1 — Backend:**
```bash
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Frontend:**
```bash
cd frontend
pnpm dev    # hoặc: npm run dev
```

### 5. Truy cập

| Service | URL |
|---------|-----|
| 🌐 Web UI | http://localhost:3000 |
| 📡 API Backend | http://localhost:8000 |
| 📖 API Docs (Swagger) | http://localhost:8000/docs |

---

## 🔑 Biến môi trường

| Biến | Bắt buộc | Mô tả | Giá trị mặc định |
|------|----------|-------|-------------------|
| `GROQ_API_KEY` | ✅ | API key từ [Groq Console](https://console.groq.com/keys) | — |
| `GROQ_MODEL` | ❌ | Model LLM | `llama-3.3-70b-versatile` |
| `RAG_BACKEND` | ❌ | Backend cho RAG | `qdrant` |
| `QDRANT_URL` | ❌ | URL Qdrant server | `http://localhost:6333` |
| `QDRANT_COLLECTION` | ❌ | Tên collection | `academy_docs` |
| `REACT_MAX_ITERATIONS` | ❌ | Số vòng ReAct tối đa | `10` |
| `REDIS_URL` | ❌ | Redis URL (cho Docker mode) | `redis://localhost:6379` |

---

## 📡 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/api/chat` | Gửi tin nhắn, nhận response (sync) |
| `POST` | `/api/chat/stream` | Gửi tin nhắn, nhận SSE stream |
| `GET` | `/api/chat/suggestions` | Gợi ý câu hỏi từ courses.csv |
| `POST` | `/api/chat/clear` | Xóa lịch sử session |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/tools` | Danh sách tools có sẵn |
| `GET` | `/api/sessions/{id}/history` | Xem lịch sử chat |
| `POST` | `/api/rag/reingest` | Nạp lại tài liệu RAG |

---

## 🧪 Kịch bản Test

Gửi lần lượt các câu sau vào chatbot để test toàn bộ pipeline:

| # | Câu test | Tính năng kiểm tra | Kỳ vọng |
|---|----------|--------------------|---------|
| 1 | "Chào bạn, bạn tên gì?" | Direct Answer | Bot chào lại, trace 1 bước |
| 2 | "Lộ trình học tại AI Academy?" | RAG | Badge RAG, trace 2 bước |
| 3 | "Tìm các khóa học Python" | Tool: course_search | Badge course_search, liệt kê khóa học |
| 4 | "Tổng học phí 3 khóa Python?" | Tool: calculator + Memory | Badge calculator, nhớ ngữ cảnh |
| 5 | "Xu hướng AI hot nhất 2025?" | Tool: web_search | Badge web_search, có thông tin từ internet |
| 6 | "So sánh Python cơ bản vs nâng cao (dạng bảng)" | Markdown Table | Bảng render đẹp trên UI |
| 7 | "Khóa Python đầu tiên có gì trong chương trình?" | Conversation Memory | Bot nhớ ngữ cảnh câu trước |

---

## 🛠 Công nghệ sử dụng

### Backend
| Công nghệ | Vai trò |
|------------|---------|
| **FastAPI** | Web framework (async, SSE) |
| **LangChain** | LLM abstraction layer |
| **LangGraph** | Agentic workflow orchestration |
| **Groq** | LLM API (Llama 3.3 70B) |
| **Qdrant** | Vector database cho RAG |
| **Sentence Transformers** | Embedding model (all-MiniLM-L6-v2) |
| **DuckDuckGo Search** | Web search API (miễn phí) |
| **Pydantic** | Data validation & settings |
| **Loguru** | Structured logging |

### Frontend
| Công nghệ | Vai trò |
|------------|---------|
| **Next.js 16** | React framework (App Router) |
| **TypeScript** | Type safety |
| **Tailwind CSS** | Utility-first CSS |
| **shadcn/ui** | UI component library |
| **ReactMarkdown** | Markdown rendering |
| **remark-gfm** | GitHub Flavored Markdown (tables) |

### DevOps (Optional)
| Công nghệ | Vai trò |
|------------|---------|
| **Docker** | Containerization |
| **Docker Compose** | Multi-service orchestration |
| **Redis** | Session memory persistence |

---

## 📝 License

MIT License — Tự do sử dụng cho mục đích học tập và phát triển.