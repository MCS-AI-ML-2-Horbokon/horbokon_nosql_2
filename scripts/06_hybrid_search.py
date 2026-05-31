from transformers import AriaTextConfig
import os
import math
import numpy as np
import pandas as pd
import re
import torch
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


load_dotenv()

INPUT_PARQUET = "./data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "./embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10
COLUMN_TITLE = "title"
COLUMN_ABSTRACT = "abstract"
COLUMN_SCORE = "score"
TEST_QUERY_1 = "BERT fine-tuning"
TEST_QUERY_2 = "Yann LeCun convolutional networks"
TEST_QUERY_3 = "making computers understand human emotions from text"
TEST_QUERY = "Image Attributes for Human Identification Protocols"

def get_words(text: str):
    return re.findall(r"\b(\w+)\b", text.lower())

# load data
print(f"Loading data: {INPUT_PARQUET}")
arxiv = pd.read_parquet(INPUT_PARQUET)
arxiv_corpus = arxiv.apply(lambda x: get_words(x[COLUMN_TITLE]) + get_words(x[COLUMN_ABSTRACT]), axis=1)
arxiv_bm25 = BM25Okapi(arxiv_corpus)

print(f"Loading model: {MODEL_NAME}")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_args = { "torch_dtype": "bfloat16" }
model = SentenceTransformer(MODEL_NAME, model_kwargs=model_args).to(device)
embeddings = np.load(INPUT_EMBEDDINGS)

def search_bm25(query: str, top_k: int = TOP_K) -> pd.DataFrame:
    words = get_words(query)
    scores = arxiv_bm25.get_scores(words)
    indexes = np.argsort(scores)[::-1][:top_k]
    results = arxiv.iloc[indexes].copy()
    results[COLUMN_SCORE] = scores[indexes]
    return results

def search_semantic_local(query: str, top_k: int = TOP_K) -> pd.DataFrame:
    embedding = model.encode(query)
    similarity = model.similarity(embedding, embeddings)
    indexes = torch.argsort(similarity, dim=-1, descending=True)[0, :top_k]
    results = arxiv.iloc[indexes].copy()
    results[COLUMN_SCORE] = similarity[0, indexes]
    return results

def RRF(results: list[pd.DataFrame], K: int = 60) -> pd.DataFrame:
    scores = {}
    rows = {}

    for result in results:
        assert COLUMN_SCORE in result.columns
        for rank, (index, row) in enumerate(result.iterrows(), start=1):
            scores[index] = scores.get(index, 0) + 1 / (K + rank)
            if index not in rows:
                rows[index] = row.drop(labels=[COLUMN_SCORE])

    fused = pd.DataFrame.from_dict(rows, orient="index")
    fused[COLUMN_SCORE] = pd.Series(scores)
    return fused.sort_values(COLUMN_SCORE, ascending=False)


def search_hybrid_local(query: str, top_k: int = TOP_K) -> pd.DataFrame:
    bm25_results = search_bm25(query, top_k)
    semantic_results = search_semantic_local(query, top_k)
    fused_results = RRF([bm25_results, semantic_results])
    return fused_results.iloc[:top_k]

print("\n=== Search with local BM25 index ===\n")
print(search_bm25(TEST_QUERY))
print("\n=== Search with local vector search ===\n")
print(search_semantic_local(TEST_QUERY))
print("\n=== Search with local hybrid search (RRF of BM25 and vector search) ===\n")
print(search_hybrid_local(TEST_QUERY))

if False:
    # load index
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.index(INDEX_NAME)
