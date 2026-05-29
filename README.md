## Частина 1 — Підготовка даних і вибір інструментів

Мета: завантажити датасет, вибрати і отримати ембеддинги.

### 1.1. Завантаження і підготовка датасету

```
PS C:\Homework\horbokon_nosql_2> uv run .\scripts\01_prepare_data.py
Читаємо датасет: 10000 it [00:00, 156415.17it/s]

Завантажено статей: 10000

Розподіл за категоріями (топ-10):
category
astro-ph              1838
hep-th                 680
hep-ph                 671
quant-ph               564
gr-qc                  350
cond-mat.mes-hall      307
cond-mat.str-el        292
cond-mat.mtrl-sci      291
cond-mat.stat-mech     271
math.AG                209
Name: count, dtype: int64

Розподіл за роками:
year
2007    10000
Name: count, dtype: int64

Приклад запису:
{'id': '0704.0001', 'title': 'Calculation of prompt diphoton production cross sections at Tevatron and   LHC energies', 'abstract': 'A fully differential calculation in perturbative quantum chromodynamics is presented for the production of massive photon pairs at hadron colliders. All next-to-leading order perturbative contributions from quark-antiquark, gluon-(anti)quark, and gluon-gluon subprocesses are included, as well as all-orders resummation of initial-state gluon radiation valid at next-to-next-to-leading logarithmic accuracy. The region of phase space is specified in which the calculation is most reliable. Good agreement is demonstrated with data from the Fermilab Tevatron, and predictions are made for more detailed tests with CDF and DO data. Predictions are shown for distributions of diphoton pairs produced at the energy of the Large Hadron Collider (LHC). Distributions of the diphoton pairs from the decay of a Higgs boson are contrasted with those produced from QCD processes at the LHC, showing that enhanced sensitivity to the signal can be obtained with judicious selection of events.', 'authors': 'BalázsC., BergerE. L., NadolskyP. M., YuanC. -P.', 'year': 2007, 'category': 'hep-ph'}

Збережено в data/arxiv_subset.parquet
```

### 1.2. Вибір інструментів

У цьому завданні використовується Pinecone як векторна база даних і allenai/specter2_base як модель ембеддингів.

У README письмово дайте відповідь на такі запитання (не менше абзацу на кожне):

1. Чим Pinecone відрізняється від Qdrant і Chroma за моделлю розгортання, ліцензією і продуктивністю? У якому сценарії ви б обрали кожен із них?
2. Чому для задачі пошуку по науковим текстам обрана модель specter2_base, а не універсальна all-MiniLM-L6-v2? Знайдіть картку моделі на HuggingFace і процитуйте, для яких задач вона навчена.
3. Що написано у картці моделі про рекомендовану метрику схожості? Чому це важливо при створенні індексу?

`TODO`


### 1.3. Отримання ембеддингів

```
PS C:\Homework\horbokon_nosql_2> uv run .\scripts\02_embed.py
Loading model: allenai/specter2_base
Loading weights: 100%|████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 72059.61it/s]
Loading data: ./data/arxiv_subset.parquet
Starting embedding: 10000 sentences
Batches: 100%|███████████████████████████████████████████████████████████████████████| 313/313 [00:28<00:00, 11.05it/s]
Completed embedding: 28.35 seconds
Total sentences embedded: 10000
Embedding dimensions: 768
L2-norm of 1st vector: 1.0
Saved to: ./embeddings/embeddings.npy
```
