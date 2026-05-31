
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

У цьому завданні використовується Pinecone як векторна база даних і `allenai/specter2_base` як модель ембеддингів.

**Питання:** Чим Pinecone відрізняється від Qdrant і Chroma за моделлю розгортання, ліцензією і продуктивністю? У якому сценарії ви б обрали кожен із них?

**Відповідь:** Pinecone - керована хмарна векторна БД: її зручно брати, коли потрібні SLA, масштабування й мінімум DevOps. Qdrant - open-source/self-hosted або cloud-рішення, краще підходить, коли важливий контроль над інфраструктурою та даними. Chroma - простіша локальна/open-source БД, зручна для прототипів, ноутбуків і невеликих RAG-проєктів, але не для високонавантаженого production.

**Питання:** Чому для задачі пошуку по науковим текстам обрана модель `specter2_base`, а не універсальна all-MiniLM-L6-v2? Знайдіть картку моделі на HuggingFace і процитуйте, для яких задач вона навчена.

**Відповідь:** `allenai/specter2_base` обрана тому, що вона навчена саме для наукових документів, а не для загальних речень. У картці моделі зазначено, що SPECTER2 призначена для задач наукового пошуку, рекомендації цитувань, класифікації документів і близьких задач із paper embeddings. `all-MiniLM-L6-v2` універсальніша й швидша, але гірше враховує специфіку наукових назв, анотацій і цитатного контексту.

**Питання:** Що написано у картці моделі про рекомендовану метрику схожості? Чому це важливо при створенні індексу?

**Відповідь:** Для SPECTER/SPECTER2 зазвичай використовують cosine similarity для порівняння ембеддингів. Це важливо, бо метрика індексу в Pinecone має відповідати способу навчання й нормалізації векторів: неправильна метрика може змінити ранжування результатів.

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

**Питання:** Поясніть, чому при використанні нормалізованих ембеддингів (одиничної довжини) косинусна схожість (cosine similarity) еквівалентна скалярному добутку (dot product)?

**Відповідь:** Для двох векторів `a` і `b` cosine similarity дорівнює `(a · b) / (||a|| ||b||)`. Якщо ембеддинги нормалізовані, то `||a|| = 1` і `||b|| = 1`, тому формула спрощується до `a · b`. Отже, для одиничних векторів cosine similarity і dot product дають однакове значення.

## Частина 2 - Завантаження даних і метадані

```
PS C:\Homework\horbokon_nosql_2> uv run .\scripts\03_load_to_pinecone.py
Upserting vectors to Pinecone: 10000 total
Upserting: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 79/79 [00:22<00:00,  3.55batch/s]
Upsert completed: 10000 vectors, 768 dimensions, cosine metric
```

## Частина 3 - Пошукові запити

```
PS C:\Homework\horbokon_nosql_2> uv run .\scripts\04_search.py
Loading model: allenai/specter2_base
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 59657.39it/s]
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

=== Local cosine similarity top-5 ===

             id                                              title                                           abstract  ...  year         category     score
378   0704.0379                        Capturing knots in polymers   This paper visualizes a knot reduction algorithm  ...  2007    cond-mat.soft  0.829367
3350  0704.3351  Symbolic sensors : one solution to the numeric...  This paper introduces the concept of symbolic ...  ...  2007  physics.ins-det  0.826005
4115  0705.0113                                    The Mathematics  This is an essay that considering the knowledg...  ...  2007          math.HO  0.825377
610   0704.0611  Modeling the field of laser welding melt pool ...  Efficient control of a laser welding process r...  ...  2007  physics.comp-ph  0.818052
3181  0704.3182  Python for Education: Computational Methods fo...  We describe a novel, interdisciplinary, comput...  ...  2007          nlin.CD  0.814229

[5 rows x 7 columns]

=== Local dot product top-5 ===

             id                                              title                                           abstract  ...  year         category     score
378   0704.0379                        Capturing knots in polymers   This paper visualizes a knot reduction algorithm  ...  2007    cond-mat.soft  0.829367
3350  0704.3351  Symbolic sensors : one solution to the numeric...  This paper introduces the concept of symbolic ...  ...  2007  physics.ins-det  0.826005
4115  0705.0113                                    The Mathematics  This is an essay that considering the knowledg...  ...  2007          math.HO  0.825377
610   0704.0611  Modeling the field of laser welding melt pool ...  Efficient control of a laser welding process r...  ...  2007  physics.comp-ph  0.818052
3181  0704.3182  Python for Education: Computational Methods fo...  We describe a novel, interdisciplinary, comput...  ...  2007          nlin.CD  0.814229

[5 rows x 7 columns]

=== Local L2 distance top-5 ===

             id                                              title                                           abstract  ...  year         category     score
378   0704.0379                        Capturing knots in polymers   This paper visualizes a knot reduction algorithm  ...  2007    cond-mat.soft  0.584180
3350  0704.3351  Symbolic sensors : one solution to the numeric...  This paper introduces the concept of symbolic ...  ...  2007  physics.ins-det  0.589906
4115  0705.0113                                    The Mathematics  This is an essay that considering the knowledg...  ...  2007          math.HO  0.590970
610   0704.0611  Modeling the field of laser welding melt pool ...  Efficient control of a laser welding process r...  ...  2007  physics.comp-ph  0.603239
3181  0704.3182  Python for Education: Computational Methods fo...  We describe a novel, interdisciplinary, comput...  ...  2007          nlin.CD  0.609542

[5 rows x 7 columns]
=== Pinecone filtered search: reinforcement learning, last 5 years, cs.LG ===

Empty DataFrame
Columns: []
Index: []

=== Pinecone filtered search: older articles before 2015 ===

             id                                              title                                           abstract  ...  year         category     score
378   0704.0379                        Capturing knots in polymers   This paper visualizes a knot reduction algorithm  ...  2007    cond-mat.soft  0.828767
3350  0704.3351  Symbolic sensors : one solution to the numeric...  This paper introduces the concept of symbolic ...  ...  2007  physics.ins-det  0.826274
4115  0705.0113                                    The Mathematics  This is an essay that considering the knowledg...  ...  2007          math.HO  0.825566
610   0704.0611  Modeling the field of laser welding melt pool ...  Efficient control of a laser welding process r...  ...  2007  physics.comp-ph  0.817017
2240  0704.2241  Why should anyone care about computing with an...  In this article we present a pedagogical intro...  ...  2007         quant-ph  0.814620
```

У файлі README надайте відповіді на обов’язкові теоретичні запитання:

**Питання:** Чи збігаються топ-5 для cosine і dot product і чому?

**Відповідь:** Для нормалізованих ембеддингів топ-5 для cosine similarity і dot product мають збігатися, бо норми всіх векторів дорівнюють 1, а cosine зводиться до скалярного добутку.

**Питання:** Чи відрізняються результати для L2 і чому?

**Відповідь:** L2-distance для одиничних векторів тісно пов’язана з cosine: `||a-b||² = 2 - 2cos(a,b)`, тому ранжування часто буде тим самим, але порядок може відрізнятися через числові похибки.

**Питання:** Що сталося б, якби ембеддинги не були нормалізовані?

**Відповідь:** Якби ембеддинги не були нормалізовані, dot product почав би враховувати не лише напрям, а й довжину вектора, тому результати могли б відрізнятися від cosine.

## Частина 4 - Chunking

```
PS C:\Homework\horbokon_nosql_2> uv run .\scripts\05_chunking.py
Loading model: allenai/specter2_base
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 49807.05it/s]
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (521 > 512). Running this sequence through the model will result in indexing errors
Upserting chunk vectors to 'arxiv-papers-fixed-chunks': 138 total
Upserting: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.73s/batch]
Upsert completed: 138 vectors, 768 dimensions, cosine metric
Upserting chunk vectors to 'arxiv-papers-semantic-chunks': 132 total
Upserting: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.28batch/s]
Upsert completed: 132 vectors, 768 dimensions, cosine metric

=== Chunk search query: teaching machines to recognize objects in pictures ===

--- Fixed-size chunks (arxiv-papers-fixed-chunks) ---

                                                    title            category  ...                                              chunk     score
2195-4  Absolute Calibration and Characterization of t...            astro-ph  ...                                    : ga detectors.  0.803441
5934-0  Spin Effects in Quantum Chromodynamics and Rec...  cond-mat.stat-mech  ...  in this thesis, we consider some spin effects ...  0.785494
987-4        Evidence for a Massive Protocluster in S255N            astro-ph  ...  ##n is forming a cluster of intermediate to hi...  0.782132
4167-4  Is Modified Gravity Required by Observations? ...            astro-ph  ...    contour for constraints from all the data sets.  0.781195
5208-4  The SSS phase of RS Ophiuchi observed with Cha...            astro-ph  ...                                      cm ^ { - 3 }.  0.778162

[5 rows x 6 columns]

--- Semantic chunks (arxiv-papers-semantic-chunks) ---

                                                    title            category  ...                                              chunk     score
9202-5  The Kinematics of the Ultra-Faint Milky Way Sa...            astro-ph  ...                                [slightly abridged]  0.790410
5934-0  Spin Effects in Quantum Chromodynamics and Rec...  cond-mat.stat-mech  ...  In this thesis, we consider some spin effects ...  0.783547
3067-0  Ages for illustrative field stars using gyroch...            astro-ph  ...  We here develop an improved way of using a rot...  0.766451
5934-1  Spin Effects in Quantum Chromodynamics and Rec...  cond-mat.stat-mech  ...  First, we analyze the helix-coil phase transit...  0.761051
656-4   The Boundary Conditions of the Heliosphere: Ph...            astro-ph  ...  These results appear to be robust since accept...  0.757879

[5 rows x 6 columns]

=== Chunk search query: reinforcement learning algorithms ===

--- Fixed-size chunks (arxiv-papers-fixed-chunks) ---

                                                    title            category  ...                                              chunk     score
2195-4  Absolute Calibration and Characterization of t...            astro-ph  ...                                    : ga detectors.  0.859494
5208-4  The SSS phase of RS Ophiuchi observed with Cha...            astro-ph  ...                                      cm ^ { - 3 }.  0.812597
2234-4  Swift observations of GRB 060614: an anomalous...            astro-ph  ...                            liso - ep correlations.  0.798420
5934-0  Spin Effects in Quantum Chromodynamics and Rec...  cond-mat.stat-mech  ...  in this thesis, we consider some spin effects ...  0.785525
4167-4  Is Modified Gravity Required by Observations? ...            astro-ph  ...    contour for constraints from all the data sets.  0.779498

[5 rows x 6 columns]

--- Semantic chunks (arxiv-papers-semantic-chunks) ---

                                                    title            category  ...                                              chunk     score
9202-5  The Kinematics of the Ultra-Faint Milky Way Sa...            astro-ph  ...                                [slightly abridged]  0.798621
5934-0  Spin Effects in Quantum Chromodynamics and Rec...  cond-mat.stat-mech  ...  In this thesis, we consider some spin effects ...  0.793027
3099-4  The Origin of the Galaxy Mass-Metallicity Rela...            astro-ph  ...  The tight observed MZR scatter is ensured when...  0.759079
8719-5  Improved constraints on dark energy from Chand...            astro-ph  ...  The small systematic scatter and tight constra...  0.756651
656-4   The Boundary Conditions of the Heliosphere: Ph...            astro-ph  ...  These results appear to be robust since accept...  0.747525

[5 rows x 6 columns]

=== Chunk search query: quantum information protocols ===

--- Fixed-size chunks (arxiv-papers-fixed-chunks) ---

                                                    title            category  ...                                              chunk     score
2195-4  Absolute Calibration and Characterization of t...            astro-ph  ...                                    : ga detectors.  0.804962
5208-4  The SSS phase of RS Ophiuchi observed with Cha...            astro-ph  ...                                      cm ^ { - 3 }.  0.800857
798-4   Spin Evolution of Accreting Neutron Stars: Non...            astro-ph  ...      be detected by advanced ligo interferometers.  0.797983
2234-4  Swift observations of GRB 060614: an anomalous...            astro-ph  ...                            liso - ep correlations.  0.785344
5934-0  Spin Effects in Quantum Chromodynamics and Rec...  cond-mat.stat-mech  ...  in this thesis, we consider some spin effects ...  0.777919

[5 rows x 6 columns]

--- Semantic chunks (arxiv-papers-semantic-chunks) ---

                                                    title            category  ...                                              chunk     score
9202-5  The Kinematics of the Ultra-Faint Milky Way Sa...            astro-ph  ...                                [slightly abridged]  0.808683
5934-0  Spin Effects in Quantum Chromodynamics and Rec...  cond-mat.stat-mech  ...  In this thesis, we consider some spin effects ...  0.792208
8719-5  Improved constraints on dark energy from Chand...            astro-ph  ...  The small systematic scatter and tight constra...  0.776630
5934-3  Spin Effects in Quantum Chromodynamics and Rec...  cond-mat.stat-mech  ...  It is shown that, contrary to the basic gluon-...  0.758138
3099-4  The Origin of the Galaxy Mass-Metallicity Rela...            astro-ph  ...  The tight observed MZR scatter is ensured when...  0.752648

[5 rows x 6 columns]
```

**Питання:** Яка стратегія дає більш осмислені чанки?

**Відповідь:** Семантичний chunking зазвичай дає більш осмислені чанки, бо не розриває речення і зберігає локальний контекст.

**Питання:** Чи є випадки розрізаних речень і як це впливає на ембеддинги?

**Відповідь:** У fixed-size chunking речення можуть обрізатися посередині, через що ембеддинг чанка гірше відображає зміст. Semantic chunking прибирає цю проблему, бо мержить повні речення.

**Питання:** Як розмір overlap впливає на кількість чанків і покриття тексту?

**Відповідь:** Більший overlap покращує покриття тексту на межах чанків, але збільшує кількість чанків, обсяг індексу і вартість пошуку.

## Частина 5 - Гібридний пошук

```
PS C:\Homework\horbokon_nosql_2> uv run .\scripts\06_hybrid_search.py
Loading data: ./data/arxiv_subset.parquet
Loading model: allenai/specter2_base
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 52241.75it/s]

=== Query: BERT fine-tuning ===

--- Top BM25 results ---

             id                                              title                                           abstract  ...  year        category      score
6243  0705.2241                       A New Measure of Fine Tuning  The solution to fine tuning is one of the prin...  ...  2007          hep-ph  19.148449
8389  0705.4387  The NMSSM Solution to the Fine-Tuning Problem,...  We present an extended study of how the Next t...  ...  2007          hep-ph  17.533773
6984  0705.2982           Fine-Tuning in Brane-antibrane Inflation  I give a brief overview of brane-antibrane inf...  ...  2007          hep-th  16.741732
8717  0706.0031  Natural SUSY Dark Matter: A Window on the GUT ...  One of the key motivations for supersymmetry i...  ...  2007          hep-ph  13.430936
3658  0704.3659  Conformal dynamics in gauge theories via non-p...  The dynamics at the IR fixed point realized in...  ...  2007          hep-ph  12.460110
3118  0704.3119  Stability and hierarchy problems in string ins...  We generalise the RS braneworld model by takin...  ...  2007          hep-th  12.460110
5571  0705.1569  Dynamical 3-Space: Supernovae and the Hubble E...  We apply the new dynamics of 3-space to cosmol...  ...  2007  physics.gen-ph  11.586302
2569  0704.2570  Inverse Monte-Carlo determination of effective...  This paper concludes our efforts in describing...  ...  2007         hep-lat  11.123798
4269  0705.0267                   Eternal Inflation is "Expensive"  The discovery of the string theory landscape h...  ...  2007          hep-th  11.044829
4385  0705.0383  String tension and removal of lattice coarseni...  We study the computation of the static quark p...  ...  2007         hep-lat  10.666228

[10 rows x 7 columns]

--- Top vector search results in Pinecone ---

             id                                              title                                           abstract  ...  year            category     score
6406  0705.2404  Misere quotients for impartial games: Suppleme...  We provide supplementary appendices to the pap...  ...  2007             math.CO  0.864534
2535  0704.2536  Introduction to Phase Transitions in Random Op...  Notes of the lectures delivered in Les Houches...  ...  2007  cond-mat.stat-mech  0.853272
6795  0705.2793    Abstract Convexity and Cone-Vexing Abstractions  This talk is a write-up on some origins of abs...  ...  2007             math.FA  0.850035
8935  0706.0249  The Compositions of the Differential Operation...  In this paper we determine the number of the m...  ...  2007             math.CO  0.848104
4441  0705.0439  Experimental local realism tests without fair ...  Following the theoretical suggestion of Ref. [...  ...  2007            quant-ph  0.847274
8125  0705.4123                            The Call of Mathematics  A few remarks on how mathematics quests for fr...  ...  2007             math.GM  0.845632
3048  0704.3049  Extracting falsifiable predictions from sloppy...  Successful predictions are among the most comp...  ...  2007            q-bio.QM  0.842980
4162  0705.0160                       Fluctuation-enhanced sensing  We present a short survey on fluctuation-enhan...  ...  2007      physics.gen-ph  0.841895
6099  0705.2097  A simple algorithm based on fluctuations to pl...  In Biology, all motor enzymes operate on the s...  ...  2007            q-fin.PM  0.839292
2138  0704.2139                   Why only few are so successful ?  In many professons employees are rewarded acco...  ...  2007      physics.pop-ph  0.836557

[10 rows x 7 columns]

--- Top hybrid search results with RRF ---

             id                                              title                                           abstract  ...  year            category     score
6243  0705.2241                       A New Measure of Fine Tuning  The solution to fine tuning is one of the prin...  ...  2007              hep-ph  0.016393
6406  0705.2404  Misere quotients for impartial games: Suppleme...  We provide supplementary appendices to the pap...  ...  2007             math.CO  0.016393
2535  0704.2536  Introduction to Phase Transitions in Random Op...  Notes of the lectures delivered in Les Houches...  ...  2007  cond-mat.stat-mech  0.016129
8389  0705.4387  The NMSSM Solution to the Fine-Tuning Problem,...  We present an extended study of how the Next t...  ...  2007              hep-ph  0.016129
6795  0705.2793    Abstract Convexity and Cone-Vexing Abstractions  This talk is a write-up on some origins of abs...  ...  2007             math.FA  0.015873
6984  0705.2982           Fine-Tuning in Brane-antibrane Inflation  I give a brief overview of brane-antibrane inf...  ...  2007              hep-th  0.015873
8717  0706.0031  Natural SUSY Dark Matter: A Window on the GUT ...  One of the key motivations for supersymmetry i...  ...  2007              hep-ph  0.015625
8935  0706.0249  The Compositions of the Differential Operation...  In this paper we determine the number of the m...  ...  2007             math.CO  0.015625
3658  0704.3659  Conformal dynamics in gauge theories via non-p...  The dynamics at the IR fixed point realized in...  ...  2007              hep-ph  0.015385
4441  0705.0439  Experimental local realism tests without fair ...  Following the theoretical suggestion of Ref. [...  ...  2007            quant-ph  0.015385

[10 rows x 7 columns]

=== Query: Yann LeCun convolutional networks ===

--- Top BM25 results ---

             id                                              title                                           abstract  ...  year            category      score
281   0704.0282  On Punctured Pragmatic Space-Time Codes in Blo...  This paper considers the use of punctured conv...  ...  2007               cs.IT  13.507847
1410  0704.1411  Trellis-Coded Quantization Based on Maximum-Ha...  Most design approaches for trellis-coded quant...  ...  2007               cs.IT  13.179666
1848  0704.1849  Response of degree-correlated scale-free netwo...  The response of degree-correlated scale-free a...  ...  2007     cond-mat.dis-nn   8.104230
1143  0704.1144                  Optimization in Gradient Networks  Gradient networks can be used to model the dom...  ...  2007  cond-mat.stat-mech   7.965842
391   0704.0392  Simulation of Robustness against Lesions of Co...  Structure entails function and thus a structur...  ...  2007            q-bio.NC   7.770103
7217  0705.3215                 On Automorphism Groups of Networks  We consider the size and structure of the auto...  ...  2007      physics.soc-ph   7.636457
196   0704.0197  Analysis of random Boolean networks using the ...  In this work we consider random Boolean networ...  ...  2007             nlin.CG   7.554429
3701  0704.3702          Statistical mechanics of complex networks  The science of complex networks is a new inter...  ...  2007  cond-mat.stat-mech   7.541949
7375  0705.3373  Laplacian Spectrum and Protein-Protein Interac...  From the spectral plot of the (normalized) gra...  ...  2007            q-bio.QM   7.513341
2950  0704.2951               Recursive weighted treelike networks  We propose a geometric growth model for weight...  ...  2007      physics.soc-ph   7.503463

[10 rows x 7 columns]

--- Top vector search results in Pinecone ---

             id                                              title                                           abstract  ...  year            category     score
4213  0705.0211  Multilayer Perceptron with Functional Inputs: ...  Functional data analysis is a growing research...  ...  2007             math.ST  0.847941
4821  0705.0819                     The Netsukuku network topology  In this document, we describe the fractal stru...  ...  2007               cs.NI  0.843090
8935  0706.0249  The Compositions of the Differential Operation...  In this paper we determine the number of the m...  ...  2007             math.CO  0.842907
610   0704.0611  Modeling the field of laser welding melt pool ...  Efficient control of a laser welding process r...  ...  2007     physics.comp-ph  0.834603
7372  0705.3370  Adaptive classification of temporal signals in...  We address the important theoretical question ...  ...  2007             math.OC  0.831383
1143  0704.1144                  Optimization in Gradient Networks  Gradient networks can be used to model the dom...  ...  2007  cond-mat.stat-mech  0.828518
6406  0705.2404  Misere quotients for impartial games: Suppleme...  We provide supplementary appendices to the pap...  ...  2007             math.CO  0.827574
6013  0705.2011        Multi-Dimensional Recurrent Neural Networks  Recurrent neural networks (RNNs) have proved e...  ...  2007               cs.AI  0.827241
4819  0705.0817                    Quantum Shortest Path Netsukuku  This document describes the QSPN, the routing ...  ...  2007               cs.NI  0.827122
1012  0704.1013                       Flops connect minimal models  A remark on a paper by Birkar-Cascini-Hacon-Mc...  ...  2007             math.AG  0.827038

[10 rows x 7 columns]

--- Top hybrid search results with RRF ---

             id                                              title                                           abstract  ...  year            category     score
1143  0704.1144                  Optimization in Gradient Networks  Gradient networks can be used to model the dom...  ...  2007  cond-mat.stat-mech  0.030777
4213  0705.0211  Multilayer Perceptron with Functional Inputs: ...  Functional data analysis is a growing research...  ...  2007             math.ST  0.016393
281   0704.0282  On Punctured Pragmatic Space-Time Codes in Blo...  This paper considers the use of punctured conv...  ...  2007               cs.IT  0.016393
4821  0705.0819                     The Netsukuku network topology  In this document, we describe the fractal stru...  ...  2007               cs.NI  0.016129
1410  0704.1411  Trellis-Coded Quantization Based on Maximum-Ha...  Most design approaches for trellis-coded quant...  ...  2007               cs.IT  0.016129
1848  0704.1849  Response of degree-correlated scale-free netwo...  The response of degree-correlated scale-free a...  ...  2007     cond-mat.dis-nn  0.015873
8935  0706.0249  The Compositions of the Differential Operation...  In this paper we determine the number of the m...  ...  2007             math.CO  0.015873
610   0704.0611  Modeling the field of laser welding melt pool ...  Efficient control of a laser welding process r...  ...  2007     physics.comp-ph  0.015625
7372  0705.3370  Adaptive classification of temporal signals in...  We address the important theoretical question ...  ...  2007             math.OC  0.015385
391   0704.0392  Simulation of Robustness against Lesions of Co...  Structure entails function and thus a structur...  ...  2007            q-bio.NC  0.015385

[10 rows x 7 columns]

=== Query: making computers understand human emotions from text ===

--- Top BM25 results ---

             id                                              title                                           abstract  ...  year         category      score
3664  0704.3665  On the Development of Text Input Method - Less...  Intelligent Input Methods (IM) are essential f...  ...  2007            cs.CL  21.880403
3661  0704.3662  An Automated Evaluation Metric for Chinese Tex...  In this paper, we propose an automated evaluat...  ...  2007            cs.HC  17.086485
7897  0705.3895  Towards Understanding the Origin of Genetic La...  Molecular biology is a nanotechnology that wor...  ...  2007         q-bio.GN  16.722691
7321  0705.3319           Detecting anchoring in financial markets  Anchoring is a term used in psychology to desc...  ...  2007         q-fin.TR  12.184513
3710  0704.3711  Maximal C*-algebras of quotients and injective...  A new C*-enlargement of a C*-algebra $A$ neste...  ...  2007          math.OA  11.884394
8443  0705.4441                          Philosophy and Relativity  With his General Theory of Relativity, Albert ...  ...  2007  physics.hist-ph  11.857198
8305  0705.4303         Database Manipulation on Quantum Computers  Manipulating a database system on a quantum co...  ...  2007         quant-ph  11.512110
2294  0704.2295  Using Image Attributes for Human Identificatio...  A secure human identification protocol aims at...  ...  2007            cs.CR  11.430138
8195  0705.4193         Lecture notes on Optical Quantum Computing  A quantum computer is a machine that can perfo...  ...  2007         quant-ph  11.350372
1266  0704.1267  Text Line Segmentation of Historical Documents...  There is a huge amount of historical documents...  ...  2007            cs.CV  10.763673

[10 rows x 7 columns]

--- Top vector search results in Pinecone ---

             id                                              title                                           abstract  ...  year        category     score
4893  0705.0891                  Opinion Dynamics and Sociophysics  No abstract given. Contents:   I. Definition a...  ...  2007  physics.soc-ph  0.828713
3664  0704.3665  On the Development of Text Input Method - Less...  Intelligent Input Methods (IM) are essential f...  ...  2007           cs.CL  0.822822
5681  0705.1679  Extracting the hierarchical organization of co...  Extracting understanding from the growing ``se...  ...  2007  physics.soc-ph  0.809153
1157  0704.1158                   Novelty and Collective Attention  The subject of collective attention is central...  ...  2007           cs.CY  0.802751
2541  0704.2542           Narratives within immersive technologies  The main goal of this project is to research t...  ...  2007           cs.HC  0.802056
9981  0706.1295       Reaction Time of a Group of Physics Students  The reaction time of a group of students major...  ...  2007   physics.ed-ph  0.801351
2082  0704.2083  Introduction to Arabic Speech Recognition Usin...  In this paper Arabic was investigated from the...  ...  2007           cs.CL  0.801183
6230  0705.2228                  The Answer is Blowing in the Wind           A 'News & Views' article -- no abstract.  ...  2007        astro-ph  0.801150
9813  0706.1127  Redesigning Computer-based Learning Environmen...  In the field of evaluation research, computer ...  ...  2007           cs.CY  0.800081
4753  0705.0751                      Approximate textual retrieval  An approximate textual retrieval algorithm for...  ...  2007           cs.IR  0.799753

[10 rows x 7 columns]

--- Top hybrid search results with RRF ---

             id                                              title                                           abstract  ...  year        category     score
3664  0704.3665  On the Development of Text Input Method - Less...  Intelligent Input Methods (IM) are essential f...  ...  2007           cs.CL  0.032522
4893  0705.0891                  Opinion Dynamics and Sociophysics  No abstract given. Contents:   I. Definition a...  ...  2007  physics.soc-ph  0.016393
3661  0704.3662  An Automated Evaluation Metric for Chinese Tex...  In this paper, we propose an automated evaluat...  ...  2007           cs.HC  0.016129
7897  0705.3895  Towards Understanding the Origin of Genetic La...  Molecular biology is a nanotechnology that wor...  ...  2007        q-bio.GN  0.015873
5681  0705.1679  Extracting the hierarchical organization of co...  Extracting understanding from the growing ``se...  ...  2007  physics.soc-ph  0.015873
1157  0704.1158                   Novelty and Collective Attention  The subject of collective attention is central...  ...  2007           cs.CY  0.015625
7321  0705.3319           Detecting anchoring in financial markets  Anchoring is a term used in psychology to desc...  ...  2007        q-fin.TR  0.015625
3710  0704.3711  Maximal C*-algebras of quotients and injective...  A new C*-enlargement of a C*-algebra $A$ neste...  ...  2007         math.OA  0.015385
2541  0704.2542           Narratives within immersive technologies  The main goal of this project is to research t...  ...  2007           cs.HC  0.015385
9981  0706.1295       Reaction Time of a Group of Physics Students  The reaction time of a group of students major...  ...  2007   physics.ed-ph  0.015152

[10 rows x 7 columns]
```

Порівняльна таблиця методів пошуку:

| Запит | BM25 top-1 | Векторний пошук top-1 | Гібридний пошук top-1 |
|---|---|---|---|
| `BERT fine-tuning` | `A New Measure of Fine Tuning` | `Misere quotients for impartial games: Supplementary material` | `A New Measure of Fine Tuning` |
| `Yann LeCun convolutional networks` | `On Punctured Pragmatic Space-Time Codes in Block Fading Channels` | `Multilayer Perceptron with Functional Inputs: An Inverse Regression Approach` | `Optimization in Gradient Networks` |
| `making computers understand human emotions from text` | `On the Development of Text Input Method - Less is More` | `Opinion Dynamics and Sociophysics` | `On the Development of Text Input Method - Less is More` |

**Питання:** Який метод дав кращий результат і чому?

**Відповідь:** У наведеному прикладі BM25 добре знаходить документ із точним збігом термінів у назві, а векторний пошук додає семантично близькі документи без обов’язкового точного збігу слів. Гібридний пошук через RRF поєднує обидва результати, тому релевантний документ, який високо стоїть в обох списках, отримує найкращий підсумковий ранг.

**Питання:** Чи є документи в топ-5 гібридного пошуку, яких немає в топ-5 окремих методів, і чому?

**Відповідь:** Такі документи можуть бути, бо RRF враховує позиції в обох ранжованих списках. Документ може не бути дуже високим в одному методі, але отримати добрий сумарний бал завдяки присутності в обох списках.

**Питання:** Як зміна параметра k в RRF впливає на видачу (наприклад, k=60 vs k=1)?

**Відповідь:** При меншому `k` сильніше впливають перші позиції, тому вищі документи отримують перевагу. При `k=60` ранги згладжуються, і різниця між сусідніми позиціями стає меншою.

## Частина 6 - Аналіз і висновки

**Питання 1:** Семантичний пошук vs BM25. Наведіть конкретні приклади запитів із вашої роботи, де кожен метод виграв. Сформулюйте загальне правило: для яких типів запитів варто надати перевагу кожному з них?

**Відповідь:** Семантичний пошук краще працює для перефразувань і запитів без точних термінів, наприклад коли зміст описано іншими словами. BM25 краще працює для точних назв, абревіатур, авторів і термінів на кшталт `BERT fine-tuning`. Загальне правило: BM25 варто обирати для лексичних збігів, а векторний пошук - для пошуку за змістом.

**Питання 2:** Вплив розміру чанка. Що відбувається з якістю пошуку, якщо чанк занадто маленький (10–15 слів)? Якщо занадто великий (500+ слів)? Чи є оптимальний розмір або він залежить від задачі?

**Відповідь:** Якщо чанк занадто маленький, наприклад 10–15 слів, він втрачає контекст і ембеддинг стає нестабільним. Якщо чанк занадто великий, наприклад 500+ слів, у ньому змішується багато тем і релевантний фрагмент розмивається. Оптимальний розмір залежить від задачі, але зазвичай потрібен баланс між повнотою контексту і точністю фрагмента.

**Питання 3:** Невідповідна метрика. Що сталося б, якби ми створили індекс Pinecone з метрикою euclidean (L2), але використовували модель, яка повертає нормалізовані вектори? Обґрунтуйте відповідь математично: виведіть зв’язок між L2 і cosine для одиничних векторів.

**Відповідь:** Для одиничних векторів `||a-b||² = ||a||² + ||b||² - 2a·b = 2 - 2cos(a,b)`. Тому для нормалізованих ембеддингів euclidean і cosine математично пов’язані та часто дають однакове ранжування, але індекс краще створювати з метрикою, рекомендованою для моделі. Невідповідна метрика може дати інші результати, особливо якщо вектори не нормалізовані.

**Питання 4:** Обмеження Pinecone Starter. З якими обмеженнями безкоштовного тіру ви зіткнулися (або могли б зіткнутися)? Як би ви вирішили задачу, якби датасет був не 10000, а 10 мільйонів статей?

**Відповідь:** У Pinecone Starter можна зіткнутися з обмеженнями на кількість індексів, обсяг векторів, регіони, продуктивність і ліміти запитів. Для 10 мільйонів статей я б використовував платну версію або іншу базу даних, можна було б створювати окремі індекси і неймспейси, але це забагато мороки заради такої задачі, краще використовувати підходящі інструменти ніж шукати обхідні рішення.
