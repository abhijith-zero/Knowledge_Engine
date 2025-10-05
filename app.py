# app.py

import streamlit as st
import json
from streamlit.components.v1 import html
from utils.query_kg import ask_question

# --- Page Config ---
st.set_page_config(
    page_title="Research Knowledge Graph Q&A",
    layout="wide",
    page_icon="🧬"
)

# --- Title ---
st.title("🧠 Research Knowledge Graph Explorer")
st.markdown("Interact with your extracted triplets, knowledge graph, and LLM-based Q&A system.")

# --- Load Data ---
try:
    with open("triplets_output.json", "r", encoding="utf-8") as f:
        triplets = json.load(f)
    st.success(f"Loaded {len(triplets)} structured triplets from local filesystem.")
except FileNotFoundError:
    st.error("❌ triplets_output.json not found. Please run `extract_triplets.py` first.")
    st.stop()

# --- Show Graph ---
st.header("🌐 Knowledge Graph")

try:
    with open("outputs/knowledge_graph.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    html(html_content, height=600)
except FileNotFoundError:
    st.error("⚠️ Knowledge Graph HTML not found. Please run `build_kg.py` first.")

# --- Q&A Chat Interface ---
st.header("💬 Ask a Question about the Research Paper")

query = st.text_input("Enter your question:")
if st.button("Ask"):
    if query.strip():
        with st.spinner("Searching and reasoning..."):
            answer = ask_question(query)
        st.subheader("🧩 Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question before clicking Ask.")

# --- Keyword and Summary Sections ---
st.header("🔑 Extracted Keywords and Summaries")

col1, col2 = st.columns(2)

with col1:
    try:
        with open("data/keywords.json", "r", encoding="utf-8") as f:
            keywords = json.load(f)
        st.write("**Top Extracted Keywords:**")
        st.write(", ".join(keywords[:20]))
    except FileNotFoundError:
        st.warning("keywords.json not found. Run `extract_keywords.py`.")

with col2:
    try:
        with open("data/summarized_triplets.json", "r", encoding="utf-8") as f:
            summaries = json.load(f)
        st.write("**Sample Summaries:**")
        for s in summaries[:3]:
            st.markdown(f"🧾 {s.get('summary', 'No summary available')}")
    except FileNotFoundError:
        st.warning("summarized_triplets.json not found. Run `summarize_triplets.py`.")
