import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Free local embedding model — 384-dim output, matches the SQL table
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def load_and_chunk_pdfs(folder="data"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    all_chunks = []
    for path in glob.glob(f"{folder}/*.pdf"):
        pages = PyPDFLoader(path).load()
        chunks = splitter.split_documents(pages)
        for c in chunks:
            all_chunks.append({
                "content": c.page_content,
                "metadata": {"source": os.path.basename(path), "page": c.metadata.get("page", 0)}
            })
    return all_chunks

def main():
    chunks = load_and_chunk_pdfs()
    print(f"Loaded {len(chunks)} chunks. Embedding + uploading...")

    texts = [c["content"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    rows = []
    for chunk, embedding in zip(chunks, embeddings):
        rows.append({
            "content": chunk["content"],
            "metadata": chunk["metadata"],
            "embedding": embedding
        })

    # Insert in batches of 50 to stay safely under request size limits
    for i in range(0, len(rows), 50):
        batch = rows[i:i+50]
        supabase.table("documents").insert(batch).execute()
        print(f"Uploaded {i + len(batch)}/{len(rows)}")

    print("Done. Your documents are live in Supabase.")

if __name__ == "__main__":
    main()