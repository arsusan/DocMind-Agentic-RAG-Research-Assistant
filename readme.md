# 📄 DocMind — Agentic RAG Research Assistant

**Live demo:** [https://docmind-agentic-rag-research-assistant-nwvgyxpbacrjcfnyvjkwvj.streamlit.app/]

DocMind is a research assistant that answers questions about your uploaded documents. Unlike a standard RAG (Retrieval-Augmented Generation) pipeline, which searches the knowledge base on _every_ query regardless of need, DocMind uses a **LangGraph agent** that decides for itself — per question — whether a document search is actually necessary. If you ask something general, it just answers. If you ask something the documents can help with, it retrieves the relevant chunks from a live Postgres vector database and grounds its answer in them, citing the source file.

That decision loop is what makes this _agentic_ RAG rather than plain RAG.

---

## Why this project

Built to demonstrate applied skills across:

- **Python** — the full ingestion pipeline and agent orchestration logic
- **SQL** — a hand-written Postgres function (`match_documents`) using `pgvector` for cosine-similarity search, running live in Supabase
- **Generative AI / LLMs** — retrieval-augmented generation with grounded, cited answers
- **Agentic AI** — a LangGraph state graph with a conditional edge, letting the LLM route between "answer directly" and "call the retrieval tool" on its own

---

## Architecture

```
                     ┌─────────────────────────┐
   PDF upload  ───▶  │   INGESTION SCRIPT       │
                     │  (chunk + embed + store) │
                     └───────────┬──────────────┘
                                 │ writes rows (content, embedding vector)
                                 ▼
                     ┌─────────────────────────┐
                     │  SUPABASE (Postgres +    │
                     │  pgvector extension)     │
                     └───────────┬──────────────┘
                                 │ similarity search (SQL function)
                                 ▼
User question ──▶ ┌─────────────────────────────┐
                   │   LANGGRAPH AGENT            │
                   │  Node 1: LLM decides —       │
                   │   "do I need to search docs, │
                   │    or answer directly?"      │
                   │  Node 2: retrieval tool runs  │
                   │   the SQL search, returns     │
                   │   chunks                      │
                   │  Loop back to Node 1 to write │
                   │   the final grounded answer   │
                   └───────────┬───────────────────┘
                                 │
                                 ▼
                     ┌─────────────────────────┐
                     │   STREAMLIT UI            │
                     │   (chat interface)        │
                     └─────────────────────────┘
```

---

## Tech stack

| Layer           | Tool                                                          | Notes                                       |
| --------------- | ------------------------------------------------------------- | ------------------------------------------- |
| Vector database | [Supabase](https://supabase.com) (Postgres + `pgvector`)      | Live, hosted, free tier                     |
| Embeddings      | `sentence-transformers` (`all-MiniLM-L6-v2`)                  | Runs locally, no API key needed, 384-dim    |
| LLM             | [Groq](https://console.groq.com) (`openai/gpt-oss-20b`)  | Free tier, fast inference                   |
| Orchestration   | [LangGraph](https://github.com/langchain-ai/langgraph)        | Conditional agent graph, not a linear chain |
| UI + hosting    | [Streamlit](https://streamlit.io) + Streamlit Community Cloud | Free live deployment                        |

**Cost to run: $0.** No paid API keys required anywhere in this stack.

---

## Project structure

```
docmind-rag/
├── data/              # PDFs to ingest go here
├── ingest.py          # chunk + embed + upload documents to Supabase
├── agent.py           # LangGraph agent definition
├── app.py             # Streamlit chat UI
├── requirements.txt
├── .env               # local secrets (not committed)
└── README.md
```

---

## How it works

1. **Ingestion** (`ingest.py`) — loads PDFs from `data/`, splits them into ~800-character overlapping chunks, embeds each chunk locally with `sentence-transformers`, and uploads the chunks + embeddings into a `documents` table in Supabase.
2. **Retrieval** — a Postgres function (`match_documents`) uses pgvector's cosine-distance operator (`<=>`) to find the most similar chunks to a query embedding.
3. **Agent** (`agent.py`) — a LangGraph `StateGraph` with two nodes: `agent` (the LLM, bound to a `search_documents` tool) and `tools` (executes the retrieval). A conditional edge (`tools_condition`) routes to the tool node only if the LLM decides it needs to search; otherwise the graph ends immediately with a direct answer.
4. **UI** (`app.py`) — a Streamlit chat interface that calls the agent and renders the conversation.

---

## Running it locally

### 1. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/docmind-rag.git
cd docmind-rag
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up Supabase

- Create a free project at [supabase.com](https://supabase.com).
- In the SQL Editor, run:

```sql
create extension if not exists vector;

create table documents (
  id bigserial primary key,
  content text,
  metadata jsonb,
  embedding vector(384)
);

create or replace function match_documents (
  query_embedding vector(384),
  match_count int default 4
)
returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;
```

### 3. Add your credentials

Create a `.env` file in the project root:

```
SUPABASE_URL=https://xxxxxxxx.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
GROQ_API_KEY=your-groq-key
```

Get a free Groq key at [console.groq.com](https://console.groq.com) → API Keys.

### 4. Ingest your documents

Drop PDFs into `data/`, then:

```bash
python ingest.py
```

### 5. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

---

## Deployment

Deployed on **Streamlit Community Cloud**:

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select this repo → main file `app.py`.
3. Add `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, and `GROQ_API_KEY` under **Advanced settings → Secrets**.
4. Deploy.

---

## Example usage

**Q: "What programming languages are listed in this resume?"**
→ Agent calls `search_documents`, retrieves the relevant chunk, answers with the source file cited.

**Q: "What can you help me with?"**
→ Agent answers directly — no document search triggered, since none is needed.

That behavioral switch — searching only when it's actually useful — is the core "agentic" design decision in this project.

---

## Future improvements

- Support for more file types (`.docx`, `.txt`, web pages)
- Multi-document filtering / source selection in the UI
- Conversation memory across sessions
- A "grading" node that checks retrieved chunks are actually relevant before answering, and re-searches if not

---
