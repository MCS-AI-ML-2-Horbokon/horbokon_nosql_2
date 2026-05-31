import os
import re
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, Vector
from sentence_transformers import SentenceTransformer

load_dotenv()

INPUT_PARQUET = "./data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "./embeddings/embeddings.npy"
INDEX_NAME_FIXED = "arxiv-papers-fixed-chunks"
INDEX_NAME_SEMANTIC = "arxiv-papers-semantic-chunks"
MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
BATCH_SIZE = 128
MAX_CONCURRENCY = 8
COLUMN_ID = "id"
COLUMN_TITLE = "title"
COLUMN_ABSTRACT = "abstract"
COLUMN_ABSTRACT_CHUNKS = "abstract_chunks"
COLUMN_ABSTRACT_LEN = "abstract_length"
COLUMN_AUTHORS = "authors"
COLUMN_CATEGORY = "category"
COLUMN_YEAR = "year"
TOKENS_IN_CHUNK = 120
TOKENS_IN_CHUNK_OVERLAP = 20
SENTENCE_SIMILARITY = 0.7 # Merge into one chunk if above
METADATA_CHUNK = "chunk"
METADATA_CHUNK_NUMBER = "chunk_number"

print(f"Loading model: {MODEL_NAME}")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME).to(device)

def count_tokens(text: str) -> int:
    return len(model.tokenizer.encode(text, add_special_tokens=False, truncation=False))

def get_fixed_size_chunks(
    text: str,
    tokens_in_chunk: int = TOKENS_IN_CHUNK,
    tokens_overlap: int = TOKENS_IN_CHUNK_OVERLAP) -> list[str]:

    tokens = model.tokenizer.encode(text, add_special_tokens=False, truncation=False)
    chunks = []
    step = max(1, tokens_in_chunk - tokens_overlap)

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + tokens_in_chunk]
        chunks.append(model.tokenizer.decode(chunk_tokens))

    return chunks

def get_semantic_chunks(
    text: str,
    tokens_in_chunk: int = TOKENS_IN_CHUNK,
    similarity_threshold: float = SENTENCE_SIMILARITY) -> list[str]:

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    if not sentences:
        return []

    embeddings = model.encode(sentences, normalize_embeddings=True)
    chunks = []
    current_sentences = [sentences[0]]
    current_tokens = count_tokens(sentences[0])
    merge_count = 0

    for i in range(1, len(sentences)):
        sentence = sentences[i]
        sentence_tokens = count_tokens(sentence)
        similarity = np.dot(embeddings[i - 1], embeddings[i])

        if (similarity < similarity_threshold or current_tokens + sentence_tokens > tokens_in_chunk):
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_tokens = sentence_tokens
            continue

        merge_count += 1
        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks

arxiv = pd.read_parquet(INPUT_PARQUET)
arxiv[COLUMN_ABSTRACT_LEN] = arxiv[COLUMN_ABSTRACT].str.len()
arxiv.sort_values(by=COLUMN_ABSTRACT_LEN, inplace=True, ascending=False)
arxiv_longest = arxiv[:30].copy()

# Upsert to Pinecone

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

for index_name in [INDEX_NAME_FIXED, INDEX_NAME_SEMANTIC]:
    if pc.has_index(index_name):
        continue

    pc.create_index(
        name=index_name,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

def upsert_chunked_index(index_name: str, chunk_function) -> None:
    vectors = []

    for id, document in arxiv_longest.iterrows():
        chunks = chunk_function(document[COLUMN_ABSTRACT])
        embeddings = model.encode(chunks, normalize_embeddings=True)

        for chunk_number, chunk in enumerate(chunks):
            vector = Vector(
                id=f"{id}-{chunk_number}",
                values=embeddings[chunk_number, :].tolist(),
                metadata={
                    COLUMN_ID: document[COLUMN_ID],
                    COLUMN_TITLE: document[COLUMN_TITLE],
                    COLUMN_AUTHORS: document[COLUMN_AUTHORS],
                    COLUMN_CATEGORY: document[COLUMN_CATEGORY],
                    COLUMN_YEAR: document[COLUMN_YEAR],
                    METADATA_CHUNK: chunk,
                    METADATA_CHUNK_NUMBER: chunk_number
                }
            )
            vectors.append(vector)

    print(f"Upserting chunk vectors to '{index_name}': {len(vectors)} total")
    index = pc.index(index_name)
    index.upsert(
        vectors=vectors,
        batch_size=BATCH_SIZE,
        max_concurrency=MAX_CONCURRENCY
    )

    stats = index.describe_index_stats()
    print(f"Upsert completed: {stats.total_vector_count} vectors, {stats.dimension} dimensions, {stats.metric} metric")

upsert_chunked_index(INDEX_NAME_FIXED, get_fixed_size_chunks)
upsert_chunked_index(INDEX_NAME_SEMANTIC, get_semantic_chunks)
