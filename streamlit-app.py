import streamlit as st
import os
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from ctransformers import AutoModelForCausalLM
from duckduckgo_search import DDGS

# --- Load Quantized Model (.gguf) ---
model_path = "./qwen_finetuned_merged_q4.gguf"

model = AutoModelForCausalLM.from_pretrained(model_path, model_type="qwen2.5")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5")

# --- Load Sentence Transformer for Embeddings ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Initialize ChromaDB ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("rag_ai_research")

# --- Streamlit App ---
st.set_page_config(page_title="AI Research RAG", layout="wide")

st.title("📄 AI Research RAG - Qwen 2.5")
st.write("A retrieval-augmented chatbot that answers questions based on recent AI research papers, blogs, and documents.")

query = st.text_input("🔍 Ask a question about AI research:", "")

# --- Function: Retrieve Relevant Documents ---
def retrieve_docs(query, k=3):
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0] if results["documents"] else []

# --- Function: Web Search for AI Papers ---
def web_search(query, max_results=3):
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query + " AI research paper", max_results=max_results)]
    return [f"{r['title']}: {r['body']}" for r in results]

# --- Function: Generate Response using Qwen 2.5 ---
def generate_response(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if query:
    # Retrieve from Local ChromaDB
    local_docs = retrieve_docs(query, k=3)
    local_context = " ".join(local_docs) if local_docs else "No local documents found."

    # Retrieve from Web Search
    web_contexts = web_search(query, max_results=3)
    web_context = " ".join(web_contexts) if web_contexts else "No web results found."

    # Combine Contexts & Generate Response
    full_context = f"Local: {local_context}\nWeb: {web_context}"
    response = generate_response(query, full_context)

    # --- Display Results ---
    st.subheader("📚 Retrieved Research Papers & Blogs")
    for doc in local_docs:
        st.markdown(f"✅ {doc}")

    st.subheader("🌐 Web AI Research Results")
    for web_doc in web_contexts:
        st.markdown(f"🔗 {web_doc}")

    st.subheader("💡 AI-Powered Response")
    st.write(response)
