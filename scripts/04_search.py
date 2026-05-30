import os
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


load_dotenv()

INPUT_PARQUET = "./data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "./embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5
COLUMN_ID = "id"
COLUMN_TITLE = "title"
COLUMN_ABSTRACT = "abstract"
COLUMN_AUTHORS = "authors"
COLUMN_CATEGORY = "category"
COLUMN_YEAR = "year"

# load model
print(f"Loading model: {MODEL_NAME}")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_args = { "torch_dtype": "bfloat16" }
model = SentenceTransformer(MODEL_NAME, model_kwargs=model_args).to(device)

# load data
print(f"Loading data: {INPUT_PARQUET}")
arxiv = pd.read_parquet(INPUT_PARQUET)

# load index
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.index(INDEX_NAME)

# load local embeddings
embeddings = np.load(INPUT_EMBEDDINGS)

def search_local(query: str, top_k: int = TOP_K):
    embedding = model.encode(query)
    similarity = model.similarity(embedding, embeddings)
    indexes = torch.argsort(similarity, dim=-1, descending=True)[0, :top_k]
    results = arxiv.iloc[indexes].copy()
    results["score"] = similarity[0, indexes]
    print(results)

def search_index(query: str, top_k: int = TOP_K):
    embedding = model.encode(query)
    results = index.query(
        vector=embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    for match in results.matches:
        meta = match.metadata
        assert meta
        print(
            f"{match.id:>15}  "
            f"{meta[COLUMN_TITLE][:49]:>49}  "
            f"{meta[COLUMN_ABSTRACT][:49]:>49}  "
            f"{meta[COLUMN_AUTHORS][:38]:>38}  "
            f"{meta[COLUMN_YEAR]}  "
            f"{meta[COLUMN_CATEGORY]:>15}  "
            f"{match.score:.6f}"
        )

print("\n=== Search in local memory ===\n")
search_local("teaching machines to recognize objects in pictures")

print("\n=== Search in Pinecone index ===\n")
search_index("teaching machines to recognize objects in pictures")
