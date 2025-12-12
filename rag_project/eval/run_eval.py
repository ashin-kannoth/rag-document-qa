import json
import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall
)

API_URL = "https://didactic-umbrella-97x5p54jwpp4hqpv-8000.app.github.dev/"
# Load your evaluation dataset
with open("rag_project/eval/eval_dataset.json") as f:
    data = json.load(f)

questions = []
answers = []
contexts = []
ground_truths = []

for item in data:
    q = item["question"]

    # Call your FastAPI /ask endpoint
    resp = requests.post(
        f"{API_URL}/ask",
        json={"question": q}
    ).json()

    answer = resp.get("answer", "")

    # Add to arrays for ragas
    questions.append(q)
    answers.append(answer)
    contexts.append([item["expected_context"]])
    ground_truths.append(item["ground_truth"])

# Ragas dataset format
dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
})

# Run evaluation
result = evaluate(
    dataset,
    metrics=[
        answer_relevancy,
        answer_correctness,
        context_precision,
        context_recall
    ]
)

print("\n===== Evaluation Results =====")
print(result)
result.to_pandas().to_csv("rag_results.csv", index=False)
print("\nSaved report to rag_results.csv")
