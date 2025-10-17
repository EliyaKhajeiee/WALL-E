# walle_rag.py
# -*- coding: utf-8 -*-
import sys
import json
from pathlib import Path
from datetime import datetime
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Fix Windows encoding issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# CONFIG
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "walle_data"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
SYNC_FILE = BASE_DIR / "sync_state.json"
MODEL_NAME = "llama3"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Set to False to skip wiki files for faster startup
LOAD_WIKI_ON_STARTUP = True

DATA_DIR.mkdir(exist_ok=True, parents=True)
VECTOR_DB_DIR.mkdir(exist_ok=True, parents=True)

# init models (lazy + optimized)
print("[INIT] Initializing WALL-E components...")
llm = Ollama(model=MODEL_NAME)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectordb = Chroma(
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
)
print("[OK] Vector DB loaded (lazy mode). Ready to query.\n")

# runtime conversation buffer
conversation_history = []
UNSAVED_TURNS = 0
SUMMARIZE_AFTER_TURNS = 8  # after N turns compress into summary

# helper: flatten JSON (for your memory files)
def flatten_json_to_entries(data: dict):
    entries = []
    for k, v in data.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                entries.append(f"{k}.{subk}: {subv}")
        elif isinstance(v, list):
            for i, item in enumerate(v):
                entries.append(f"{k}.{i}: {item}")
        else:
            entries.append(f"{k}: {v}")
    return entries

def load_sync_state():
    """Load the sync state tracking which files have been loaded."""
    if SYNC_FILE.exists():
        try:
            return json.loads(SYNC_FILE.read_text(encoding="utf-8"))
        except:
            return {}
    return {}

def save_sync_state(state):
    """Save the sync state to disk."""
    SYNC_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

def load_json_files():
    """Load only new or modified JSON files into the vector DB (batched for speed)."""
    print("[LOAD] Checking for new/updated memory files...")
    sync_state = load_sync_state()
    loaded = 0
    updated = 0

    # Collect all files that need loading
    files_to_load = []
    for file in DATA_DIR.glob("*.json"):
        file_name = file.name

        # Skip wiki files if disabled
        if not LOAD_WIKI_ON_STARTUP and file_name.startswith("wiki_"):
            continue

        file_mtime = file.stat().st_mtime

        # Skip if already loaded and not modified
        if file_name in sync_state and sync_state[file_name] == file_mtime:
            continue

        files_to_load.append((file, file_name, file_mtime))

    if not files_to_load:
        print("[OK] Memory up to date. No changes detected.\n")
        return

    print(f"[LOAD] Processing {len(files_to_load)} files...")

    # Batch process files
    batch_texts = []
    batch_metadatas = []
    BATCH_SIZE = 100  # Process 100 files at a time

    for idx, (file, file_name, file_mtime) in enumerate(files_to_load):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            entries = flatten_json_to_entries(data)

            if entries:
                batch_texts.extend(entries)
                batch_metadatas.extend([{"source": file_name}] * len(entries))

                if file_name in sync_state:
                    updated += 1
                else:
                    loaded += 1

                sync_state[file_name] = file_mtime

                # Add to vector DB in batches
                if len(batch_texts) >= BATCH_SIZE or idx == len(files_to_load) - 1:
                    if batch_texts:
                        vectordb.add_texts(batch_texts, metadatas=batch_metadatas)
                        print(f"[PROGRESS] Loaded {idx + 1}/{len(files_to_load)} files...")
                        batch_texts = []
                        batch_metadatas = []

        except Exception as e:
            print(f"[WARNING] Error loading {file_name} - {e}")

    save_sync_state(sync_state)
    print(f"[OK] Memory synced: {loaded} new, {updated} updated.\n")

def save_fact(key: str, value: str):
    """Save a conversation turn to learned_data.json and vector DB."""
    target = DATA_DIR / "learned_data.json"
    try:
        d = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    except:
        d = {}
    d[key] = value
    target.write_text(json.dumps(d, indent=2), encoding="utf-8")
    vectordb.add_texts([f"{key}: {value}"], metadatas=[{"source": "learned_data.json"}])

    # Update sync state for learned_data.json
    sync_state = load_sync_state()
    sync_state["learned_data.json"] = target.stat().st_mtime
    save_sync_state(sync_state)

def summarize_and_store(conversation):
    full = "\n".join([f"You: {m['user']}\nWALL-E: {m['ai']}" for m in conversation])
    prompt = (
        "Summarize the following conversation into concise factual bullet points "
        "(1-3 sentences per fact):\n\n" + full + "\n\nSummary:"
    )
    summary = llm.invoke(prompt).strip()
    if summary:
        timestamp = datetime.utcnow().isoformat()
        vectordb.add_texts(
            [f"conversation_summary.{timestamp}: {summary}"],
            metadatas=[{"source": "conversation_summary", "ts": timestamp}],
        )
    return summary

def build_context(user_input: str):
    print("[SEARCH] Retrieving relevant info...")
    retrieved = vectordb.similarity_search(user_input, k=5)
    context = "\n".join(
        [f"- {r.page_content} (source={r.metadata.get('source')})" for r in retrieved]
    )
    print(f"[FOUND] Retrieved {len(retrieved)} relevant documents.\n")
    return context

def query_walle(user_input: str):
    global UNSAVED_TURNS
    user_input = user_input.strip()
    if not user_input:
        return
    context = build_context(user_input)
    conversation = "\n".join(
        [f"You: {m['user']}\nWALL-E: {m['ai']}" for m in conversation_history[-10:]]
    )
    prompt = f"""You are WALL-E, a friendly robot companion. Answer in 1-2 SHORT sentences maximum. Be helpful and warm, but BRIEF. No long explanations.

Facts:
{context}

Recent chat:
{conversation}

User: {user_input}
WALL-E:"""
    response = llm.invoke(prompt).strip()

    # Only add to conversation history, DON'T save every interaction as a fact
    conversation_history.append({"user": user_input, "ai": response})
    UNSAVED_TURNS += 1

    # Periodically summarize and store important conversations
    if UNSAVED_TURNS >= SUMMARIZE_AFTER_TURNS:
        summarize_and_store(conversation_history[-SUMMARIZE_AFTER_TURNS:])
        UNSAVED_TURNS = 0

    print("WALL-E:", response, "\n")

def main():
    load_json_files()
    print("[READY] WALL-E RAG ready. Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ")
        except KeyboardInterrupt:
            print("\nWALL-E: Goodbye.")
            break
        if user_input.lower() in ("exit", "quit"):
            print("WALL-E: Goodbye.")
            break
        query_walle(user_input)

if __name__ == "__main__":
    main()
