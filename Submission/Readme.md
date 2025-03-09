# RAG-based AI Research Assistant

This project implements a Retrieval-Augmented Generation (RAG) system using a local LLaMA model for inference.


### 1. **Clone the repository (if you haven’t already)**

If you haven't cloned the repository yet, you can do so by running:

```bash
git clone https://github.com/Neural-Navigators-CSI/Intellihack_Neural_Navigators_Task03.git
cd Intellihack_Neural_Navigators_Task03/Submission
```
-------------------------------------------
## Important
For the better performance, you need a gpu to run inference.py. If not you will need to run it 
on kaggel. So the notebook is provided as inference.ipynb. You just need to add langchain_docs.json(explained in the report. it needs to create chroma db) which is given in this directory as kaggle input.
### Setup Instructions

### 1. Create and Activate Virtual Environment

#### On Windows:
```sh
python -m venv venv
venv\Scripts\activate
```

#### On Mac/Linux:
```sh
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Run Inference
```sh
python inference.py
```

## Project Structure
```
project-folder/
│── inference.py         # Main script for running inference
│── requirements.txt     # Required dependencies
│── README.md            # Documentation
│── chroma_db/           # Persistent ChromaDB storage
│── langchain_docs.json  # JSON file containing indexed documents
```

## How It Works
1. Loads a **fine-tuned LLaMA model** from a GGUF file.
2. Uses **ChromaDB** to store and retrieve relevant documents.
3. Generates responses based on retrieved context.
4. Displays a progress bar for document retrieval and response generation.

## Notes
- Ensure your GGUF model path in `inference.py` is correct.
- The script will automatically retrieve and generate answers for your queries.
- You can modify `k=3` in `vector_store.similarity_search(query, k=3)` to change the number of retrieved documents.

## Example Usage
```python
query = "What is DualPipe’s overlap strategy?"
print(rag_generate(query))
```

