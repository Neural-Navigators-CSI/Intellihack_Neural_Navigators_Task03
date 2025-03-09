from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from langchain.docstore.document import Document
import os


with open("Data/documents.pkl", "rb") as f:
    documents = pickle.load(f)

env-key = os.environ["OPENAI_API_KEY"]
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)

langchain_docs = [Document(page_content=chunk) for chunk in all_chunks]
dataset = generator.generate(documents, 10, num_personas=0)

pandas_df = dataset.to_pandas()
synthetic_dataset = Dataset.from_pandas(pandas_df)

column_mapping = {
    "user_input": "question",
    "reference": "answer",
    "reference_contexts": "contexts"
}
synthetic_dataset = synthetic_dataset.rename_columns(column_mapping)

# Map synthesizer_name to question_type (simplified)
def map_question_type(synth_name):
    if "single_hop" in synth_name:
        return "simple"
    elif "multi_hop" in synth_name:
        return "reasoning"
    else:
        return "multi_context"

synthetic_dataset = synthetic_dataset.map(
    lambda x: {"question_type": map_question_type(x["synthesizer_name"])}
)
synthetic_dataset = synthetic_dataset.remove_columns(["synthesizer_name"])  # Drop old column

train_test = synthetic_dataset.train_test_split(test_size=0.2)
train_dataset = train_test["train"]
test_dataset = train_test["test"]

train_dataset.save_to_disk("Data/train-dataset")
test_dataset.save_to_disk("Data/test-dataset")

# if you need to viuslaize this on ragas dashboard
os.environ["RAGAS_APP_TOKEN"]

# it will open on localhost:3000