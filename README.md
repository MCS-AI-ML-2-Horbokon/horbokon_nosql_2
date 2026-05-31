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
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 62186.45it/s]
Loading data: ./data/arxiv_subset.parquet

=== Search in local memory ===

             id                                              title                                           abstract  ...  year         category     score
378   0704.0379                        Capturing knots in polymers   This paper visualizes a knot reduction algorithm  ...  2007    cond-mat.soft  0.829367
3350  0704.3351  Symbolic sensors : one solution to the numeric...  This paper introduces the concept of symbolic ...  ...  2007  physics.ins-det  0.826005
4115  0705.0113                                    The Mathematics  This is an essay that considering the knowledg...  ...  2007          math.HO  0.825377
610   0704.0611  Modeling the field of laser welding melt pool ...  Efficient control of a laser welding process r...  ...  2007  physics.comp-ph  0.818052
3181  0704.3182  Python for Education: Computational Methods fo...  We describe a novel, interdisciplinary, comput...  ...  2007          nlin.CD  0.814229

[5 rows x 7 columns]

=== Search in Pinecone index ===

             id                                              title                                           abstract  ...  year         category     score
378   0704.0379                        Capturing knots in polymers   This paper visualizes a knot reduction algorithm  ...  2007    cond-mat.soft  0.828767
3350  0704.3351  Symbolic sensors : one solution to the numeric...  This paper introduces the concept of symbolic ...  ...  2007  physics.ins-det  0.826274
4115  0705.0113                                    The Mathematics  This is an essay that considering the knowledg...  ...  2007          math.HO  0.825566
610   0704.0611  Modeling the field of laser welding melt pool ...  Efficient control of a laser welding process r...  ...  2007  physics.comp-ph  0.817017
2240  0704.2241  Why should anyone care about computing with an...  In this article we present a pedagogical intro...  ...  2007         quant-ph  0.814620

[5 rows x 7 columns]
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

## Частина 5 — Гібридний пошук

```
PS C:\Homework\horbokon_nosql_2> uv run .\scripts\06_hybrid_search.py
Loading data: ./data/arxiv_subset.parquet
Loading model: allenai/specter2_base
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 51890.99it/s]

=== Search with local BM25 index ===

             id                                              title                                           abstract  ...  year  category      score
2294  0704.2295  Using Image Attributes for Human Identificatio...  A secure human identification protocol aims at...  ...  2007     cs.CR  46.760870
5364  0705.1362  Studies of EGRET sources with a novel image re...  We have developed an image restoration techniq...  ...  2007  astro-ph  17.312981
8187  0705.4185  Secure Two-party Protocols for Point Inclusion...  It is well known that, in theory, the general ...  ...  2007     cs.CR  14.570931
6582  0705.2580  Quantum protocols for transference of proof of...  Zero-knowledge proof system is an important pr...  ...  2007  quant-ph  14.002364
9837  0706.1151  A taxonomic Approach to Topology Control in Ad...  Topology Control (TC) aims at tuning the topol...  ...  2007     cs.NI  13.855781
6906  0705.2904  Key rate of quantum key distribution with hash...  We propose an information reconciliation proto...  ...  2007  quant-ph  13.728585
4206  0705.0204  Using Images to create a Hierarchical Grid Spa...  This paper presents a hybrid approach to spati...  ...  2007     cs.DS  13.639945
5526  0705.1524         Studies of Cosmic Rays with GeV Gamma Rays  We describe the role of GeV gamma-ray observat...  ...  2007  astro-ph  13.542249
1736  0704.1737     Quantum memory for images - a quantum hologram  Matter-light quantum interface and quantum mem...  ...  2007  quant-ph  13.385296
3398  0704.3399  Cooperative Transmission Protocols with High S...  Cooperative transmission is an emerging commun...  ...  2007     cs.IT  13.348538

[10 rows x 7 columns]

=== Search with local vector search ===

             id                                              title                                           abstract  ...  year        category     score
2294  0704.2295  Using Image Attributes for Human Identificatio...  A secure human identification protocol aims at...  ...  2007           cs.CR  0.837673
9005  0706.0319               Even more simple cardinal invariants  Using GCH, we force the following: There are c...  ...  2007         math.LO  0.818546
4954  0705.0952  An Independent Evaluation of Subspace Face Rec...  This paper explores a comparative study of bot...  ...  2007           cs.CV  0.817667
70    0704.0071  Pairwise comparisons of typological profiles (...  No abstract given; compares pairs of languages...  ...  2007  physics.soc-ph  0.816096
7742  0705.3740                        Optimal Iris Fuzzy Sketches  Fuzzy sketches, introduced as a link between b...  ...  2007           cs.CR  0.815805
5587  0705.1585  HMM Speaker Identification Using Linear and No...  Speaker identification is a powerful, non-inva...  ...  2007           cs.LG  0.807193
6940  0705.2938  Codage arithmetique pour la description d'une ...  Using predictive adaptive arithmetic coding an...  ...  2007         stat.ME  0.804245
6303  0705.2301                                     Withrawn paper  This paper has been withdrawn by the authors d...  ...  2007           gr-qc  0.802214
4822  0705.0820  ANDNA: the distributed hostname management sys...  We present the Abnormal Netsukuku Domain Name ...  ...  2007           cs.NI  0.801659
5801  0705.1799  Subjective Questions and Answers for a Mathema...  This article of mathematical education reflect...  ...  2007         math.GM  0.798628

[10 rows x 7 columns]

=== Search with local hybrid search (RRF of BM25 and vector search) ===

             id                                              title                                           abstract  ...  year        category     score
2294  0704.2295  Using Image Attributes for Human Identificatio...  A secure human identification protocol aims at...  ...  2007           cs.CR  0.032787
5364  0705.1362  Studies of EGRET sources with a novel image re...  We have developed an image restoration techniq...  ...  2007        astro-ph  0.016129
9005  0706.0319               Even more simple cardinal invariants  Using GCH, we force the following: There are c...  ...  2007         math.LO  0.016129
8187  0705.4185  Secure Two-party Protocols for Point Inclusion...  It is well known that, in theory, the general ...  ...  2007           cs.CR  0.015873
4954  0705.0952  An Independent Evaluation of Subspace Face Rec...  This paper explores a comparative study of bot...  ...  2007           cs.CV  0.015873
70    0704.0071  Pairwise comparisons of typological profiles (...  No abstract given; compares pairs of languages...  ...  2007  physics.soc-ph  0.015625
6582  0705.2580  Quantum protocols for transference of proof of...  Zero-knowledge proof system is an important pr...  ...  2007        quant-ph  0.015625
9837  0706.1151  A taxonomic Approach to Topology Control in Ad...  Topology Control (TC) aims at tuning the topol...  ...  2007           cs.NI  0.015385
7742  0705.3740                        Optimal Iris Fuzzy Sketches  Fuzzy sketches, introduced as a link between b...  ...  2007           cs.CR  0.015385
5587  0705.1585  HMM Speaker Identification Using Linear and No...  Speaker identification is a powerful, non-inva...  ...  2007           cs.LG  0.015152

[10 rows x 7 columns]

=== Search with vector search in Pinecone ===

             id                                              title                                           abstract  ...  year        category     score
2294  0704.2295  Using Image Attributes for Human Identificatio...  A secure human identification protocol aims at...  ...  2007           cs.CR  0.837298
4954  0705.0952  An Independent Evaluation of Subspace Face Rec...  This paper explores a comparative study of bot...  ...  2007           cs.CV  0.818065
9005  0706.0319               Even more simple cardinal invariants  Using GCH, we force the following: There are c...  ...  2007         math.LO  0.817518
70    0704.0071  Pairwise comparisons of typological profiles (...  No abstract given; compares pairs of languages...  ...  2007  physics.soc-ph  0.816376
7742  0705.3740                        Optimal Iris Fuzzy Sketches  Fuzzy sketches, introduced as a link between b...  ...  2007           cs.CR  0.815991
5587  0705.1585  HMM Speaker Identification Using Linear and No...  Speaker identification is a powerful, non-inva...  ...  2007           cs.LG  0.807165
6940  0705.2938  Codage arithmetique pour la description d'une ...  Using predictive adaptive arithmetic coding an...  ...  2007         stat.ME  0.804714
6303  0705.2301                                     Withrawn paper  This paper has been withdrawn by the authors d...  ...  2007           gr-qc  0.802262
4822  0705.0820  ANDNA: the distributed hostname management sys...  We present the Abnormal Netsukuku Domain Name ...  ...  2007           cs.NI  0.800605
5801  0705.1799  Subjective Questions and Answers for a Mathema...  This article of mathematical education reflect...  ...  2007         math.GM  0.799484

[10 rows x 7 columns].

=== Search with hybrid search with index in Pinecone and local BM25 ===

             id                                              title                                           abstract  ...  year        category     score
2294  0704.2295  Using Image Attributes for Human Identificatio...  A secure human identification protocol aims at...  ...  2007           cs.CR  0.032787
5364  0705.1362  Studies of EGRET sources with a novel image re...  We have developed an image restoration techniq...  ...  2007        astro-ph  0.016129
4954  0705.0952  An Independent Evaluation of Subspace Face Rec...  This paper explores a comparative study of bot...  ...  2007           cs.CV  0.016129
8187  0705.4185  Secure Two-party Protocols for Point Inclusion...  It is well known that, in theory, the general ...  ...  2007           cs.CR  0.015873
9005  0706.0319               Even more simple cardinal invariants  Using GCH, we force the following: There are c...  ...  2007         math.LO  0.015873
70    0704.0071  Pairwise comparisons of typological profiles (...  No abstract given; compares pairs of languages...  ...  2007  physics.soc-ph  0.015625
6582  0705.2580  Quantum protocols for transference of proof of...  Zero-knowledge proof system is an important pr...  ...  2007        quant-ph  0.015625
9837  0706.1151  A taxonomic Approach to Topology Control in Ad...  Topology Control (TC) aims at tuning the topol...  ...  2007           cs.NI  0.015385
7742  0705.3740                        Optimal Iris Fuzzy Sketches  Fuzzy sketches, introduced as a link between b...  ...  2007           cs.CR  0.015385
5587  0705.1585  HMM Speaker Identification Using Linear and No...  Speaker identification is a powerful, non-inva...  ...  2007           cs.LG  0.015152

[10 rows x 7 columns]
```
