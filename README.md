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

Поясніть, чому при використанні нормалізованих ембеддингів (одиничної довжини) косинусна схожість (cosine similarity) еквівалентна скалярному добутку (dot product)?

`TODO`

## Частина 2 — Завантаження даних і метадані

```
PS C:\Homework\horbokon_nosql_2> uv run .\scripts\03_load_to_pinecone.py
Upserting vectors to Pinecone: 10000 total
Upserting: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 79/79 [00:22<00:00,  3.55batch/s]
Upsert completed: 10000 vectors, 768 dimensions, cosine metric
```

## Частина 3 — Пошукові запити

```
PS C:\Homework\horbokon_nosql_2> uv run .\scripts\04_search.py
Loading model: allenai/specter2_base
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 5506.22it/s]
Loading data: ./data/arxiv_subset.parquet

=== Search in local memory ===

             id                                              title                                           abstract                                 authors  year         category     score
378   0704.0379                        Capturing knots in polymers   This paper visualizes a knot reduction algorithm            VirnauP., KardarM., KantorY.  2007    cond-mat.soft  0.828615
3350  0704.3351  Symbolic sensors : one solution to the numeric...  This paper introduces the concept of symbolic ...              BenoitEric, FoulloyLaurent  2007  physics.ins-det  0.825920
4115  0705.0113                                    The Mathematics  This is an essay that considering the knowledg...                       HJavier Guachalla  2007          math.HO  0.824928
610   0704.0611  Modeling the field of laser welding melt pool ...  Efficient control of a laser welding process r...  BracicA. Borstnik, GovekarE., GrabecI.  2007  physics.comp-ph  0.817517
3181  0704.3182  Python for Education: Computational Methods fo...  We describe a novel, interdisciplinary, comput...    MyersChristopher R., SethnaJames. P.  2007          nlin.CD  0.813897

=== Search in Pinecone index ===

      0704.0379                        Capturing knots in polymers   This paper visualizes a knot reduction algorithm            VirnauP., KardarM., KantorY.  2007    cond-mat.soft  0.827655
      0704.3351  Symbolic sensors : one solution to the numerical-  This paper introduces the concept of symbolic sen              BenoitEric, FoulloyLaurent  2007  physics.ins-det  0.826808
      0705.0113                                    The Mathematics  This is an essay that considering the knowledge s                       HJavier Guachalla  2007          math.HO  0.825229
      0704.0611  Modeling the field of laser welding melt pool by   Efficient control of a laser welding process requ  BracicA. Borstnik, GovekarE., GrabecI.  2007  physics.comp-ph  0.817434
      0704.2241  Why should anyone care about computing with anyon  In this article we present a pedagogical introduc       BrennenGavin K., PachosJiannis K.  2007         quant-ph  0.814218
```

У файлі README надайте відповіді на обов’язкові теоретичні запитання:

Чи збігаються топ-5 для cosine і dot product і чому?
Чи відрізняються результати для L2 і чому?
Що сталося б, якби ембеддинги не були нормалізовані?

`TODO`

## Частина 4 — Chunking

```
PS C:\Homework\horbokon_nosql_2> uv run .\scripts\05_chunking.py
Loading model: allenai/specter2_base
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 73982.14it/s]
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (521 > 512). Running this sequence through the model will result in indexing errors
Upserting chunk vectors to 'arxiv-papers-fixed-chunks': 138 total
Upserting: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.24batch/s]
Upsert completed: 138 vectors, 768 dimensions, cosine metric
Upserting chunk vectors to 'arxiv-papers-semantic-chunks': 132 total
Upserting: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.21batch/s]
Upsert completed: 132 vectors, 768 dimensions, cosine metric
```
