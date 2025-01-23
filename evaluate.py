from scipy import stats
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "DogaOytc/llama3-m-s"
model = AutoModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = "emrecan/stsb-mt-turkish"
eval_dataset = load_dataset(dataset, split="test")

def compute_metrics(preds, truths):
    spearman_corr = stats.spearmanr(preds, truths).correlation
    return {"spearman_correlation": spearman_corr}

def get_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_dim)
    embeddings = hidden_states.mean(dim=1)  # Mean pooling across the sequence
    return embeddings

def evaluate_model(model, tokenizer, eval_dataset):
    model.eval()
    preds = []
    truths = []

    for example in eval_dataset:
        emb1 = get_embeddings(example["sentence1"], model, tokenizer)
        emb2 = get_embeddings(example["sentence2"], model, tokenizer)

        # Compute cosine similarity
        emb1 = emb1 / emb1.norm(dim=1, keepdim=True)
        emb2 = emb2 / emb2.norm(dim=1, keepdim=True)
        cos_sim = torch.matmul(emb1, emb2.T).item()

        preds.append(cos_sim)
        truths.append(example["score"])

    # Compute metrics
    metrics = compute_metrics(preds, truths)
    print(f"Spearman Correlation: {metrics['spearman_correlation']:.4f}")
    return metrics

evaluate_model(model, tokenizer, eval_dataset)