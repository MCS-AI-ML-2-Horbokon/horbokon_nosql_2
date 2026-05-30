import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, Vector

load_dotenv()

INPUT_PARQUET = "./data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "./embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 128
MAX_CONCURRENCY = 8
COLUMN_ID = "id"
COLUMN_TITLE = "title"
COLUMN_ABSTRACT = "abstract"

# Ініціалізація клієнта
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Створюємо індекс (якщо не існує)
if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

documents = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)
vectors = [
    Vector(
        id=d[COLUMN_ID],
        values=embeddings[i, :].tolist(),
        metadata={
            "title": d[COLUMN_TITLE][:200],
            "abstract": d[COLUMN_ABSTRACT][:500]
        }
    )
    for i, d in documents.iterrows()
]

print(f"Upserting vectors to Pinecone: {len(vectors)} total")
index = pc.index(INDEX_NAME)
index.upsert(
    vectors=vectors,
    batch_size=BATCH_SIZE,
    max_concurrency=MAX_CONCURRENCY
)

stats = index.describe_index_stats()
print(f"Upsert completed: {stats.total_vector_count} vectors, {stats.dimension} dimensions, {stats.metric} metric")
