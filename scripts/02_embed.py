import numpy as np
import pandas as pd
import time
import torch
import torch.cuda
import os
from sentence_transformers import SentenceTransformer


INPUT_PATH = "./data/arxiv_subset.parquet"
OUTPUT_PATH = "./embeddings/embeddings.npy"
MODEL_NAME = "allenai/specter2_base"
COLUMN_TITLE = "title"
COLUMN_ABSTRACT = "abstract"

os.makedirs("embeddings", exist_ok=True)

# load model
print(f"Loading model: {MODEL_NAME}")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_args = {
    "torch_dtype": "bfloat16"
}
model = SentenceTransformer(MODEL_NAME, model_kwargs=model_args).to(device)

# load data
print(f"Loading data: {INPUT_PATH}")
arxiv = pd.read_parquet(INPUT_PATH)
sentences = arxiv[COLUMN_TITLE].str.cat(arxiv[COLUMN_ABSTRACT], sep=model.tokenizer.sep_token).tolist()
sentences.sort(key=len)

# embeddings
print(f"Starting embedding: {len(sentences)} sentences")
start = time.time()
embeddings = model.encode(
    sentences,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True
)
finish = time.time() - start
print(f"Completed embedding: {finish:.2f} seconds")

count, dims = embeddings.shape
print(f"Total sentences embedded: {count}")
print(f"Embedding dimensions: {dims}")
print(f"L2-norm of 1st vector: {np.linalg.norm(embeddings[0, :])}")

np.save(OUTPUT_PATH, embeddings)
print(f"Saved to: {OUTPUT_PATH}")
