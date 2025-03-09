import os
import PyPDF2
import pickle
from glob import glob
import wikipedia
from langchain.text_splitter import CharacterTextSplitter

dir_path = "Data/"
files = os.listdir(dir_path)

print("Files in the folder:", files)

files = glob(os.path.join(dir_path, "*.md")) + glob(os.path.join(dir_path, "*.pdf"))
documents = []

for file in files:
    if file.endswith(".md"):
        with open(file, "r", encoding="utf-8") as f:
            documents.append(f.read())
    elif file.endswith(".pdf"):
        with open(file, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
            documents.append(text)

print(f"Loaded {len(documents)} documents: {[os.path.basename(f) for f in files]}")

wikipedia.set_lang("en")

pages = [
    "Artificial Intelligence",
    "Natural Language Processing",
    "DeepSeek",
    "Deep Learning",
    "AI Research"
]

wiki_data = []
for page in pages:
    try:
        content = wikipedia.page(page).content
        wiki_data.append(content)
    except Exception as e:
        print(f"Error fetching {page}: {e}")

with open("Data/wiki_ai_data.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(wiki_data))

wiki_cleaned = [" ".join(doc.split()) for doc in wiki_data]
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
wiki_chunks = [chunk for doc in wiki_cleaned for chunk in text_splitter.split_text(doc)]

documents.extend(wiki_cleaned)

print("Loaded all given docs and wikipedia documents")
with open("Data/processed_documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("Documents saved to processed_documents.pkl")