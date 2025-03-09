import chromadb
from tqdm import tqdm
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from llama_cpp import Llama
import json
from huggingface_hub import hf_hub_download

repo_id = "ranaweerahk/Neural_Navigators-qwen2.5-q4-gguf"
gguf_file = "qwen_finetuned_merged_q4.gguf"
downloaded_path = hf_hub_download(repo_id=repo_id, filename=gguf_file, local_dir="./downloaded_model")

llm = Llama(model_path=downloaded_path, n_ctx=4096, verbose=False)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("rag_ai_research")

with open("langchain_docs.json", "r") as f:
    loaded_docs_dict = json.load(f)
loaded_docs = [Document(page_content=d["page_content"]) for d in loaded_docs_dict]

# Create vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(
    documents=loaded_docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

def truncate_context(context, max_chars=1000):
    return context[:max_chars] + "..." if len(context) > max_chars else context

def rag_generate(query):
    print("Retrieving documents...")
    with tqdm(total=3, desc="Retrieving", ncols=80) as pbar:
        retrieved_docs = vector_store.similarity_search(query, k=3)
        pbar.update(3)

    context = " ".join([doc.page_content for doc in retrieved_docs])
    context = truncate_context(context, max_chars=3000)

    print("Generating response...")
    with tqdm(total=1, desc="Generating", ncols=80) as pbar:
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        outputs = llm(prompt, max_tokens=200, temperature=0.7)
        pbar.update(1)

    return outputs["choices"][0]["text"]

# Test
query = "What is DualPipe’s overlap strategy?"
print(rag_generate(query))