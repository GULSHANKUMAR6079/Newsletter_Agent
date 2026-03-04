# 📝 Blog Writing Agent

> **Production-grade AI blog writing pipeline** powered by Google Gemini, LangGraph, and Streamlit.
> Researches the web, plans an outline, writes sections in parallel, generates images,
> and delivers SEO metadata + social media content — all in one click.

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-ff6b35?logo=langchain)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285f4?logo=google&logoColor=white)](https://ai.google.dev)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+pgvector-336791?logo=postgresql&logoColor=white)](https://postgresql.org)
[![CI](https://img.shields.io/github/actions/workflow/status/yourusername/blog-writing-agent/ci.yml?label=CI)](https://github.com/yourusername/blog-writing-agent/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔀 **Smart Routing** | Automatically decides between closed-book, hybrid, or open-book research modes |
| 🔍 **Web Research** | Tavily-powered search + LLM evidence synthesis |
| 🧩 **Structured Planning** | Generates a detailed 5–9 task outline with audience, tone, and constraints |
| ⚡ **Parallel Writing** | Sections written concurrently via LangGraph's Send API |
| 🖼️ **Image Generation** | Gemini-generated technical diagrams inserted into the post |
| 🎨 **SEO Metadata** | Slug, meta description, keywords, OG title, reading time |
| 📱 **Social Content** | LinkedIn post + Twitter/X thread generated automatically |
| ⭐ **Quality Review** | 5-dimension scoring rubric with publish/revise/reject verdict |
| 📚 **Session History** | PostgreSQL + pgvector for persistence and semantic search |
| 🌐 **Export** | Download as Markdown, ZIP bundle, or self-contained HTML |

---

## 🏗️ Architecture

```mermaid
graph TD
    A([START]) --> B[🔀 Router]
    B -->|needs_research=true| C[🔍 Research]
    B -->|needs_research=false| D[🧩 Orchestrator]
    C --> D
    D -->|fan-out via Send| E1[✍️ Worker 1]
    D --> E2[✍️ Worker 2]
    D --> En[✍️ Worker N]
    E1 --> F[🔗 Reducer]
    E2 --> F
    En --> F
    F --> G[🎨 SEO]
    G --> H[📱 Social]
    H --> I[⭐ Quality Reviewer]
    I --> J([END])

    subgraph Reducer Subgraph
        F1[Merge Sections] --> F2[Plan Images] --> F3[Generate Images]
    end
```

---

## 📂 Project Structure

```
blog writing agent/
├── app/
│   ├── config.py           # AppConfig dataclass (env-based)
│   ├── models.py           # All Pydantic schemas + LangGraph State
│   ├── database.py         # PostgreSQL async ORM (SQLAlchemy + pgvector)
│   └── graph/
│       ├── graph.py        # LangGraph pipeline assembly
│       ├── router.py       # Research mode decision
│       ├── research.py     # Tavily search + evidence synthesis
│       ├── orchestrator.py # Blog outline generation
│       ├── worker.py       # Parallel section writer (tenacity retry)
│       ├── reducer.py      # Merge + image generation
│       ├── seo.py          # SEO metadata generation
│       ├── social.py       # LinkedIn + Twitter content
│       └── reviewer.py     # Quality scoring rubric
├── app/utils/
│   ├── file_utils.py       # Slug, zip, HTML export, word count
│   ├── image_utils.py      # Gemini image generation helpers
│   └── logging_utils.py    # Centralised logger factory
├── frontend/
│   ├── app.py              # Streamlit UI (8 tabs)
│   └── styles.css          # Premium dark theme (Inter, glassmorphism)
├── tests/
│   ├── test_models.py      # Pydantic schema validation tests
│   ├── test_utils.py       # File utils unit tests
│   └── test_graph_nodes.py # Mocked node tests (no API calls)
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml      # App + PostgreSQL 16 + pgvector
├── Makefile
├── requirements.txt
└── .env.example
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yourusername/blog-writing-agent.git
cd "blog writing agent"
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY (required)
# Add TAVILY_API_KEY to enable web research (optional)
```

### 3. Start PostgreSQL (Docker)

```bash
docker run -d \
  --name bwa_postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=blogwriter \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 4. Run the app

```bash
streamlit run frontend/app.py
```

Open **<http://localhost:8501>** 🎉

---

## 🐳 Docker Compose (Full Stack)

```bash
# Copy and edit your .env first
cp .env.example .env

# Start everything (app + postgres)
make docker-up

# View logs
make docker-logs

# Stop
make docker-down
```

---

## 🧪 Tests

```bash
make test
# or
python -m pytest tests/ -v
```

Tests use mocked LLM calls — no API key required.

---

## ⚙️ Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | ✅ | — | Google AI Studio API key |
| `TAVILY_API_KEY` | ☑️ | — | Enables web research mode |
| `DATABASE_URL` | ☑️ | `postgresql+asyncpg://postgres:postgres@localhost:5432/blogwriter` | PostgreSQL connection string |
| `GEMINI_MODEL` | ☑️ | `gemini-2.5-flash` | LLM model name |
| `LLM_TEMPERATURE` | ☑️ | `0.7` | Generation temperature |
| `MAX_RETRIES` | ☑️ | `3` | LLM retry attempts |
| `OUTPUT_DIR` | ☑️ | `output` | Where .md files are saved |
| `MAX_IMAGES_PER_BLOG` | ☑️ | `3` | Max generated images |
| `DEBUG` | ☑️ | `false` | Verbose SQL + logging |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make changes + add tests
4. Run `make lint && make test`
5. Open a Pull Request

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

*Built with ❤️ using Google Gemini, LangGraph, Streamlit, and PostgreSQL.*
