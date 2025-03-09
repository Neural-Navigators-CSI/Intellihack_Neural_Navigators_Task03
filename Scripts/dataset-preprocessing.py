import re
import nltk
import pickle
from nltk.corpus import stopwords

with open("Data/processed_documents.pkl", "rb") as f:
    documents = pickle.load(f)

print(f"Loaded {len(documents)} documents for preprocessing.")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text

documents = [clean_text(doc) for doc in documents]

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

documents = [" ".join([word for word in doc.split() if word not in stop_words]) for doc in documents]

with open("Data/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("Documents saved to documents.pkl")