from datasets import load_from_disk
import json
from datasets import Dataset
from datasets import load_from_disk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np
from final-architecture import rag_generate


test_dataset = load_from_disk("Data/test-dataset")
print(test_dataset[0])

# Retrieval metrics
def precision_at_k(retrieved_docs, relevant_docs, k):
    retrieved_texts = [doc.page_content for doc in retrieved_docs[:k]]
    relevant_texts = [ctx for ctx in relevant_docs]
    relevant_in_retrieved = sum(1 for doc in retrieved_texts if doc in relevant_texts)
    return relevant_in_retrieved / min(k, len(retrieved_texts))

def recall_at_k(retrieved_docs, relevant_docs, k):
    retrieved_texts = [doc.page_content for doc in retrieved_docs[:k]]
    relevant_texts = [ctx for ctx in relevant_docs]
    relevant_in_retrieved = sum(1 for doc in retrieved_texts if doc in relevant_texts)
    return relevant_in_retrieved / len(relevant_texts) if relevant_texts else 0

# Generation metrics
def exact_match(predicted, reference):
    return 1 if predicted.strip() == reference.strip() else 0

def bleu_score(predicted, reference):
    return sentence_bleu([reference.split()], predicted.split())

def rouge_l(predicted, reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, predicted)
    return scores['rougeL'].fmeasure

def coherence_score(predicted):
    # Simple heuristic: length and presence of key terms
    return min(1.0, len(predicted.split()) / 10)  # Normalize to 0-1

# Combined metric
def answer_relevance(retrieval_score, generation_score):
    return 0.5 * retrieval_score + 0.5 * generation_score

def evaluate_rag(test_dataset):
    results = {
        "precision@3": [],
        "recall@3": [],
        "exact_match": [],
        "bleu": [],
        "rouge_l": [],
        "coherence": [],
        "answer_relevance": []
    }

    for example in test_dataset:
        question = example["question"]
        reference_answer = example["answer"]
        relevant_contexts = example["contexts"]

        # Generate response
        retrieved_docs, predicted_answer = rag_generate(question)

        # Retrieval metrics
        p3 = precision_at_k(retrieved_docs, relevant_contexts, 3)
        r3 = recall_at_k(retrieved_docs, relevant_contexts, 3)
        # Generation metrics
        em = exact_match(predicted_answer, reference_answer)
        bleu = bleu_score(predicted_answer, reference_answer)
        rouge = rouge_l(predicted_answer, reference_answer)
        coh = coherence_score(predicted_answer)

        retrieval_score = (p3 + r3) / 2
        generation_score = (bleu + rouge + coh) / 3
        relevance = answer_relevance(retrieval_score, generation_score)
        # Store results
        results["precision@3"].append(p3)
        results["recall@3"].append(r3)
        results["exact_match"].append(em)
        results["bleu"].append(bleu)
        results["rouge_l"].append(rouge)
        results["coherence"].append(coh)
        results["answer_relevance"].append(relevance)

    # Aggregate results
    aggregated = {metric: np.mean(scores) for metric, scores in results.items()}
    return aggregated

# Run evaluation
eval_results = evaluate_rag(test_dataset)

for metric, score in eval_results.items():
    print(f"{metric}: {score:.2f}")