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
COLUMN_SCORE = "score"

# load model
print(f"Loading model: {MODEL_NAME}")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME).to(device)

# load data
print(f"Loading data: {INPUT_PARQUET}")
arxiv = pd.read_parquet(INPUT_PARQUET)

# load index
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.index(INDEX_NAME)

# load local embeddings
embeddings = np.load(INPUT_EMBEDDINGS)

def search_local(query: str, top_k: int = TOP_K) -> pd.DataFrame:
    embedding = model.encode(query)
    similarity = model.similarity(embedding, embeddings)
    indexes = torch.argsort(similarity, dim=-1, descending=True)[0, :top_k]
    results = arxiv.iloc[indexes].copy()
    results[COLUMN_SCORE] = similarity[0, indexes]
    return results


def search_local_cosine(query: str, top_k: int = TOP_K) -> pd.DataFrame:
    embedding = model.encode(query, normalize_embeddings=True)
    scores = embeddings @ embedding
    indexes = np.argsort(scores)[::-1][:top_k]
    results = arxiv.iloc[indexes].copy()
    results[COLUMN_SCORE] = scores[indexes]
    return results


def search_local_dot_product(query: str, top_k: int = TOP_K) -> pd.DataFrame:
    embedding = model.encode(query, normalize_embeddings=True)
    scores = embeddings @ embedding
    indexes = np.argsort(scores)[::-1][:top_k]
    results = arxiv.iloc[indexes].copy()
    results[COLUMN_SCORE] = scores[indexes]
    return results


def search_local_l2(query: str, top_k: int = TOP_K) -> pd.DataFrame:
    embedding = model.encode(query, normalize_embeddings=True)
    distances = np.linalg.norm(embeddings - embedding, axis=1)
    indexes = np.argsort(distances)[:top_k]
    results = arxiv.iloc[indexes].copy()
    results[COLUMN_SCORE] = distances[indexes]
    return results

def search_index(query: str, top_k: int = TOP_K) -> pd.DataFrame:
    embedding = model.encode(query)
    results = index.query(
        vector=embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    rows = {}
    for match in results.matches:
        meta = match.metadata
        assert meta
        rows[match.id] = {
            COLUMN_ID: meta[COLUMN_ID],
            COLUMN_TITLE: meta[COLUMN_TITLE],
            COLUMN_ABSTRACT: meta[COLUMN_ABSTRACT],
            COLUMN_AUTHORS: meta[COLUMN_AUTHORS],
            COLUMN_YEAR: meta[COLUMN_YEAR],
            COLUMN_CATEGORY: meta[COLUMN_CATEGORY],
            COLUMN_SCORE: match.score
        }

    return pd.DataFrame.from_dict(rows, orient="index")

query = "teaching machines to recognize objects in pictures"

print("\n=== Search in local memory ===\n")
print(search_local(query))

print("\n=== Search in Pinecone index ===\n")
print(search_index(query))

print("\n=== Local cosine similarity top-5 ===\n")
print(search_local_cosine(query))

print("\n=== Local dot product top-5 ===\n")
print(search_local_dot_product(query))

print("\n=== Local L2 distance top-5 ===\n")
print(search_local_l2(query))
