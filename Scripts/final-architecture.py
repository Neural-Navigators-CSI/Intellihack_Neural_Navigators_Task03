from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from colbert import Indexer
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Collection
import torch
import numpy as np
from langchain.vectorstores import Chroma
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from transformers import AutoModel, AutoTokenizer
from duckduckgo_search import DDGS

model = AutoModel.from_pretrained("./qwen_finetuned")
tokenizer = AutoTokenizer.from_pretrained("./qwen_finetuned")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


vector_store = Chroma.from_documents(
    documents=langchain_docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
vector_store.persist()

# Chromadb+ model
def rag_generate(query):
    retrieved_docs = vector_store.similarity_search(query, k=3)
    context = " ".join([doc.page_content for doc in retrieved_docs])

    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


query = "What is DualPipe’s overlap strategy?"
print(rag_generate(query))

# colbert+chroma
checkpoint = "colbert-ir/colbertv2.0"
config = ColBERTConfig(
    nbits=2,
    root="./colbert_data"
)
collection = Collection(data=[doc.page_content for doc in langchain_docs])
with Run().context(RunConfig(nranks=1, experiment="rag")):  # Single GPU
    indexer = Indexer(checkpoint=checkpoint, config=config)

    indexer.index(name="rag_index", collection=collection)
def get_colbert_embeddings(docs, indexer):
    embeddings = []
    with torch.no_grad():
        for doc in docs:
            # Encode document (returns [seq_len, dim] tensor)
            doc_emb = indexer.encode([doc.page_content]).cpu().numpy()  # [1, seq_len, dim]
            # Average across tokens for a single vector
            doc_emb = np.mean(doc_emb[0], axis=0)  # [dim]
            embeddings.append(doc_emb)
    return embeddings

colbert_embeddings = get_colbert_embeddings(langchain_docs, indexer)

vector_store = Chroma.from_documents(
    documents=langchain_docs,
    embedding=None,
    collection_name="rag_colbert",
    persist_directory="./chroma_db",
    embedding_function=lambda x: colbert_embeddings
)

vector_store._collection.add(
    embeddings=colbert_embeddings,
    documents=[doc.page_content for doc in langchain_docs],
    ids=[f"doc_{i}" for i in range(len(langchain_docs))]
)

vector_store.persist()

def rag_generate(query):
    retrieved_docs = vector_store.similarity_search(query, k=3)
    context = " ".join([doc.page_content for doc in retrieved_docs])

    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
query = "What is DualPipe’s overlap strategy?"
print(rag_generate(query))


# CrossEncoder +Chroma
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=None,
    collection_name="rag_colbert"
)

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
def enhanced_retrieve(query, k=5, rerank_k=3):
    initial_docs = vector_store.similarity_search(query, k=k)
    initial_texts = [doc.page_content for doc in initial_docs]

    pairs = [[query, text] for text in initial_texts]
    scores = cross_encoder.predict(pairs)

    sorted_pairs = sorted(zip(scores, initial_docs), reverse=True)
    reranked_docs = [doc for _, doc in sorted_pairs[:rerank_k]]

    return reranked_docs


# Web-search
def web_search(query, max_results=3):
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=max_results)]
        web_contexts = [f"{r['title']}: {r['body']}" for r in results]
        return web_contexts
    except Exception as e:
        print(f"Web search failed: {e}")
        return []
def enhanced_rag_generate(query):
    # Local retrieval with reranking
    local_docs = enhanced_retrieve(query, k=5, rerank_k=3)
    local_context = " ".join([doc.page_content for doc in local_docs])

    web_contexts = web_search(query, max_results=3)
    web_context = " ".join(web_contexts)

    full_context = f"Local: {local_context}\nWeb: {web_context}" if web_context else local_context

    prompt = f"Context: {full_context}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=150)  # Increased for richer context
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return local_docs, web_contexts, response


