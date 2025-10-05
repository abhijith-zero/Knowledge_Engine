
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import pipeline

# --- Initialize embeddings model and vector DB ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# You can adjust the vector DB path
VECTOR_DB_PATH = "vector_db"

# Load the vector database (or create if missing)
def get_vector_db():
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=VECTOR_DB_PATH
    ))

    # Check if collection exists, else create
    try:
        collection = client.get_collection("papers")
    except:
        collection = client.create_collection("papers")
    return collection

# --- Function to create vector database from knowledge texts ---
def create_vector_db(knowledge_json="data/knowledge_texts.json"):
    """
    Reads knowledge_texts.json and stores embeddings in Chroma vector DB
    """
    collection = get_vector_db()
    model = SentenceTransformer(EMBEDDING_MODEL)

    with open(knowledge_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        collection.add(
            documents=[item["text"]],
            metadatas=[{"source": item.get("source", "Unknown")}],
            ids=[str(item["id"])]
        )
    print(f"✅ Vector DB created with {len(data)} entries.")

# --- Function to ask a question ---
def ask_question(query, top_k=3):
    """
    Returns an answer based on retrieved relevant knowledge chunks
    """
    # Load models
    model = SentenceTransformer(EMBEDDING_MODEL)
    llm = pipeline("text-generation", model="facebook/bart-large-cnn")

    # Retrieve top-k relevant chunks
    collection = get_vector_db()
    results = collection.query(query_texts=[query], n_results=top_k)
    retrieved_texts = [doc for doc in results["documents"][0]]

    if not retrieved_texts:
        return "Sorry, I could not find relevant information."

    # Combine retrieved context
    context = "\n".join(retrieved_texts)

    # Generate answer
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    answer = llm(prompt, max_new_tokens=150, do_sample=False)[0]["generated_text"]

    # Optional: clean answer (remove repeated context)
    answer = answer.replace(context, "").strip()
    return answer

# --- Test ---
if __name__ == "__main__":
    # create_vector_db()  # Uncomment to build DB first
    question = "What causes bone loss in microgravity?"
    print("💬 Answer:", ask_question(question))
