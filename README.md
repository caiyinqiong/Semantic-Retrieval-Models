<div align="center">
    <p><b>Awesome Semantic Models for the First-stage Retrieval</b></p>
    <br><br>
   <img width="500" src="./images/architecture.png" alt="Awesome">
</div>

> Note: 
>
> - A curated list of awesome papers for the **Semantic Retrieval**, including some early methods and recent neural models.
> - For people who want to acquire semantic models for re-ranking stage, we refer readers to the awesome NeuIR survey by *[Guo et.al](https://arxiv.org/abs/1903.06902)*.
> - Any feedback and contribution are welcome, please open an issue or [contact me](mailto:caiyinqiong18s@ict.ac.cn).

## Contents

- [Survey paper](#survey-paper)
- [Chapter 1: Classical Term-based Retrieval](#classical-term-based-etrieval)
- [Chapter 2: Early Methods for Semantic Retrieval](#early-methods-for-semantic-retrieval)
  - [Query Expansion](#query-expansion)
  - [Term Dependency Model](#term-dependency-model)
  - [Topic Model](#topic-model)
  - [Translation Model](#translation-model)
- [Chapter 3: Neural Methods for Semantic Retrieval](#neural-methods-for-semantic-retrieval)
  - [Sparse Retrieval Methods](#sparse-retrieval-methods)
  - [Dense Retrieval Methods](#dense-retrieval-methods)
  - [Hybrid of Sparse-dense Retrieval Methods](#hybrid-of-sparse-dense-retrieval-methods)
- [Chapter 4: Other Resources](#other-resources)
  - [Other Tasks](#other-tasks)
  - [Datasets](#datasets)
  - [Index](#index)

***



## Survey Paper

- [Semantic Models for the First-stage Retrieval: A Comprehensive Review.](https://arxiv.org/pdf/2103.04831.pdf) 



## Classical Term-based Retrieval

- 【VSM】A vector space model for automatic indexing（1975）
- 【TFIDF】Developments in automatic text retrieval（1991）
- 【TFIDF】Term-weighting approaches in automatic text retrieval.（1988）
- 【BIM】Relevance weighting of search terms（1976）
- 【Probabilistic Model】Relevance weighting of search terms（1977）、Probabilistic models of  indexing and searching（1981）、The uniﬁed probabilistic model for ir（1983）、Efﬁcient probabilistic inference for text retrieval（1991）、Some simple effective approximations to the 2-poisson model for probabilistic weighted retrieval（1994）、The Probability Ranking Principle in IR（1997，propose Probability Ranking Principle）、The uniﬁed model revisited（2003）、A new uniﬁed probabilistic model（2004）
- 【Tree Dependence Model】A theoretical basis for the use of co-occurrence data in information retrieval
- 【BM25】The probabilistic relevance framework: BM25 and beyond（2010）
- 【DFR】Probabilistic models of information retrieval based on measuring the divergence from randomness（2002）
- 【Language Model/QL】A language modeling approach to information retrieval（1998）
- 【Language Model】Statistical language models for information retrieval a critical review（zhai,2007）
- 【Language Model】Hypergeometric language model and Zipf-like scoring function for web document similarity retrieval（2010）



## Early Methods for Semantic Retrieval

### Query Expansion

- Global methods expand or reformulate query words by analyzing the word co-occurrences from the corpus being searched or using hand-crafted thesaurus
- Global analysis often relies on external resources or document collections
- Word-word associations in document retrieval systems.（1969）
- Concept based query expansion（1993）
- 【RM3】Relevance based language models（2001）
- Model-based feedback in the language modeling approach to information retrieval.（2001）
- Query expansion using local and global document analysis.
- 【PRF】Query expansion using lexical-semantic relations.（1994）
- 【PRF】Selecting good expansion terms for pseudo-relevance feedback.（2008）
- 【Rocchio PRF】Relevance Feedback in Information Retrieval（1971）
- Search Engines: Information Retrieval in Practice.（2009）
- A comparative study of methods for estimating query language models with pseudo feedback.（2009）
- Pseudo-relevance feedback based on matrix factorization（2016）
- Reducing the risk of query expansion via robust constrained optimization（2009，query drift problem）

### Term Dependency Model

- 【VSM + term dependency】
  - Experiments in automatic phrase indexing for document retrieval: A comparison of syntactic and non-syntactic methods（1987）
  - An analysis of statistical and syntactic phrases（1997）
  - Term-weighting approaches in automatic text retrieval（1988）
  - A probabilistic model of information retrieval: development and comparative experiments.（2000）
- 【BM25 + term dependency】
  - Relevance ranking using kernels.（2010）
- 【LM + term dependency】
  - A general language model for information retrieval.（1999）
  - Biterm language models for document retrieval.（2002）
  - Capturing term dependencies using a language model based on sentence trees.（2002）
  - Dependence language model for information retrieval（2004）
- A generative theory of relevance（2004）
- 【SDM】A Markov random ﬁeld model for term dependencies（2005）

### Topic Model

- 【GVSM】Generalized vector spaces model in information retrieval.（1985）
- 【LSA/LSI】Indexing by latent semantic analysis.（1990）
- 【NMF】Algorithms for non-negative matrix factorization.（2001）
- 【RLSI】Regularized latent semantic indexing（2011）、Group matrix factorization for scalable topic modeling.（2012）
- 【PLSA】Probabilistic latent semantic indexing（1999）
- 【LDA】Latent dirichlet allocation（2003）
- Supervised topic models.（2007）
- Investigating task performance of probabilistic topic models: an empirical study of PLSA and LDA.（2010）
- Corpus structure, language models, and ad hoc information retrieval.（2004，topic model for smoothing)
- Regularizing ad hoc retrieval scores（2005，topic model for smoothing）
- Lda-based document models for ad-hoc retrieval.（2006，LDA for IR and LDA for LM smoothing）
- A comparative study of utilizing topic models for information retrieval.（2009）
- Latent semantic indexing (lsi) fails for trec collections（2011）

### Translation Model

- Information retrieval as statistical translation.（1999）
- Estimation of statistical translation models based on mutual information for ad hoc information retrieval.（2010）
- Axiomatic analysis of translation language model for information retrieval.（2012）
- Language model for information retrieval（2002）



## Neural Methods for Semantic Retrieval
### Sparse Retrieval Methods

- **Term Re-weighting**
  - 【DeepTR】**Learning to reweight terms with distributed representations**（2015）
  - 【NTLM】**Integrating and evaluating neural word embeddings in information retrieval**（2015）
  - 【TVD】**Learning Term Discrimination**（SIGIR2020）
  - 【DeepCT】**Context-Aware Sentence/Passage Term Importance Estimation for First Stage Retrieval**（2019）
  - 【DeepCT】**Context-Aware Term Weighting For First-Stage Passage Retrieval**（SIGIR2020）
  - 【DeepCT】**Efficiency Implications of Term Weighting for Passage Retrieval**（SIGIR2020）
  - 【HDCT】**Context-Aware Document Term Weighting for Ad-Hoc Search**（WWW2020）
  - 【uniCOIL】**A Few Brief Notes on DeepImpact, COIL, and a Conceptual Framework for Information Retrieval Techniques**（2021）<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />
  
- **Expansion**
  - 【GAR】**Generation-augmented retrieval for open-domain question answering**（2020，query expansion）
  - 【Doc2Query】**Document Expansion by query Prediction**（2019）
  - 【DocTTTTTQuery】**From doc2query to docTTTTTquery**（2019）
  - 【UED】**A Unified Pretraining Framework for Passage Ranking and Expansion**（AAAI2021）
- **Expansion + Term Re-weighting**
  - 【SparTerm】**SparTerm: Learning Term-based Sparse Representation for Fast Text Retrieval**（2020）
  - 【SPLADE】**SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking**（SIGIR2021）
  - **Learning Passage Impacts for Inverted Indexes**（SIGIR2021）
- **Sparse Representation**
  - **Semantic hashing**（2009）
  - 【SNRM】**From Neural Re-Ranking to Neural Ranking: Learning a Sparse Representation for Inverted Indexing**（CIKM2018）
  - 【UHD-BERT】**UHD-BERT: Bucketed Ultra-High Dimensional Sparse Representations for Full Ranking**（2021）
  - 【BPR】**Efﬁcient Passage Retrieval with Hashing for Open-domain Question Answering**（2021）<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />

### Dense Retrieval Methods

- **Word-embedding-based**
  - 【FV】**Aggregating continuous word embeddings for information retrieval**（2013）
  - **Short Text Similarity with Word Embeddings**（CIKM2015）
  - 【DESM】**A Dual Embedding Space Model for Document Ranking**（2016）
  - **Efficient natural language response suggestion for smart reply**（2017）
  - **End-to-end retrieval in continuous space**（2018）
- 【PIQA】**Phrase indexed question answering: A new challenge for scalable document comprehension**（EMNLP2018）
- 【CoRT】**CoRT: Complementary Rankings from Transformers**（ICTIR2019）
- 【DPR】Dense Passage Retrieval for Open-Domain Question Answering（2020）
- 【RepBERT】**RepBERT: CONTEXTUALIZED TEXT EMBEDDINGS FOR FIRST-STAGE RETRIEVAL**（2020）
- 【DC-BERT】**DC-BERT: Decoupling Question and Document for Efficient Contextual Encoding**（2020）
- **Neural Retrieval for Question Answering with Cross-Attention Supervised Data Augmentation**
- 【ANCE】**Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval**
- 【RocketQA】**An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering**
- 【LTRe】**Learning to Retrieve：how to train a dense retrieval model effectively and efficiently**
- **Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling**（SIGIR2021）
- 【STAR/ADORE】**Optimizing Dense Retrieval Model Training with Hard Negatives**（SIGIR2021）
- 【DensePhrases】**Learning Dense Representations of Phrases at Scale**（2021）
- 【DANCE】**More Robust Dense Retrieval with Contrastive Dual Learning**<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />
- 【PAIR】**PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval**（2021）<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />
- 【ColBERT-QA】**Relevance-guided Supervision for OpenQA with ColBERT**（2021）<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />
- **End-to-End Training of Multi-Document Reader and Retriever for Open-Domain Question Answering**（2021）<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />
- **Distilling Knowledge from Reader to Retriever for Question Answering**<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />
- **Knowledge Distillation**
  - 【TCT-ColBERT】**Distilling Dense Representations for Ranking using Tightly-Coupled Teachers**
  - **Distilling Knowledge for Fast Retrieval-based Chat-bots**（2020）
  - **Improving Bi-encoder Document Ranking Models with Two Rankers and Multi-teacher Distillation**
  - **Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation**（2021，Margin-MSE loss）
- **Multi-vector Representation** 
  - 【MUPPET】**Multi-Hop Paragraph Retrieval for Open-Domain Question Answering**（ACL2019）
  - 【Poly-encoders】**Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring**（ICLR2020）
  - 【ME-BERT】**Sparse, Dense, and Attentional Representations for Text Retrieval**（2020）
  - 【ColBERT】**ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**（SIGIR2020）
  - 【COIL】**COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List**（2021）
  - **Improving Document Representations by Generating Pseudo Query Embeddings for Dense Retrieval**（ACL2021）
- **Accelerate Interaction-based Models**
  - **Incorporating query term independence assumption for efficient retrieval and ranking using deep neural networks**（2019）
  - **Efficient Interaction-based Neural Ranking with Locality Sensitive Hashing**（WWW2019）
  - 【PreTTR】**Efficient Document Re-Ranking for Transformers by Precomputing Term Representations**（SIGIR2020）
  - 【DeFormer】**DeFormer: Decomposing Pre-trained Transformers for Faster Question Answering**（ACL2020）
  - 【SPARTA】**SPARTA: Efﬁcient Open-Domain Question Answering via Sparse Transformer Matching Retrieval**
  - **Conformer-kernel with query term independence for document retrieval**（2020）
- **Pre-training**
  - 【ORQA】**Latent Retrieval for Weakly Supervised Open Domain Question Answering**（ACL2019）
  - 【REALM】 **Retrieval-Augmented Language Model Pre-Training**（2020）
  - **Pre-training Tasks for Embedding-based Large-scale Retrieval**（2020）
  - **Embedding-based Zero-shot Retrieval through Query Generation**（query generation）
  - **Zero-shot neural retrieval via domain-targeted synthetic query generation**（2020，query generation）
  - **Towards Robust Neural Retrieval Models with Synthetic Pre-Training**（query generation）
  - **Pre-trained Language Model for Web-scale Retrieval in Baidu Search**<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />
- **Joint Learning with Index**
  - **Joint Learning of Deep Retrieval Model and Product Quantization based Embedding Index**（SIGIR2021）
  - **Jointly Optimizing Query Encoder and Product Quantization to Improve Retrieval Performance**（2021）<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />

### Hybrid of Sparse-dense Retrieval Methods

- **Monolingual and cross-lingual information retrieval models based on (bilingual) word embeddings**（2015）
-  【GLM】 **Word embedding based generalized language model for information retrieval**（2015）
- 【BOW-CNN】**Learning Hybrid Representations to Retrieve Semantically Equivalent Questions**（2015）
- 【GLM】 **Word embedding based generalized language model for information retrieval**（2015）
- **Representing documents and queries as sets of word embedded vectors for information retrieval**（2016）
- 【DESM_mixture】**A Dual Embedding Space Model for Document Ranking**（2016）
- 【DenSPI】**Real-Time Open-Domain Question Answering with Dense-Sparse Phrase Index** ACL 2019. [paper][code]
- 【SPARC】**Contextualized Sparse Representations for Real-Time Open-Domain Question Answering**（ACL2020）
- 【EPIC】**Expansion via Prediction of Importance with Contextualization**（SIGIR2020）
- 【Hybrid】**Sparse, Dense, and Attentional Representations for Text Retrieval**（2020）
- 【CLEAR】**Complementing lexical retrieval with semantic residual embedding**（2020）
- **Leveraging Semantic and Lexical Matching to Improve the Recall of Document Retrieval Systems: A Hybrid Approach**（2020）
- **Contextualized Offline Relevance Weighting for Efficient and Effective Neural Retrieval**（SIGIR2021）<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />




## Other Resources
### Other Tasks
- **Embedding-based Retrieval in Facebook Search**（KDD2020）
- **K-way D-dimensional Discrete Codes for Compact Embedding Representations**
- **Deep Retrieval: An End-to-End Learnable Structure Model for Large-Scale Recommendations**
- **From Semantic Retrieval to Pairwise Ranking: Applying Deep Learning in E-commerce Search**（SIGIR2019，Jingdong）
- 【DPSR】**Towards Personalized and Semantic Retrieval: An End-to-End Solution for E-commerce Search via Embedding Learning**（SIGIR2020，Jingdong）
- **MOBIUS: Towards the Next Generation of Query-Ad Matching in Baidu’s Sponsored Search**（Baidu）
- 【DIN】**Deep interest network for click-through rate prediction**
- 【MIND】**Multi-Interest Network with Dynamic Routing for Recommendation at Tmall**（2019，Tmall）
- 【DMIN】**Deep Multi-Interest Network for Click-through Rate Prediction**（CIKM2020）
- **Few-Shot Conversational Dense Retrieval**（SIGIR2021，dialogue）
-  **Report on the First HIPstIR Workshop on the Future of Information Retrieval**（2019，workshop）
-  **Let’s measure run time! Extending the IR replicability infrastructure to include performance aspects**（SIGIR2019）
-  **Off the Beaten Path: Let’s Replace Term-Based Retrieval with k-NN Search**（2016CIKM）

### Datasets
- 【TREC CAR】**TREC Complex Answer Retrieval Overview**
- 【MS MARCO】**A human-generated machine reading comprehension datase**

### Index
- **Tree-based**
  - 【KD tree】**Multidimensional binary search trees used for associative searching**（1975）
- **Hashing-based**
  - 【LSH】**Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality**（1998）
  - **Asymmetric LSH for sublinear time maximum inner product search**（2014）
- **Quantization-based**
  - 【PQ】**Product quantization for nearest neighbor search**（2010）
  - 【OPQ】**Optimized product quantization**
  - **Additive quantization for extreme vector compression**（2014）
  - **Composite quantization for approximate nearest neighbor search**（2014）
  - **Tree quantization for large-scale similarity search and classification**（2015）
- **Graph-based**
  - **Scalable k-NN graph construction for visual descriptors**（2012）
  - 【NSW】**Navigation in a small world**
  - 【HNSW】**Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs**
- Toolkits
  - [Faiss: a library for efficient similarity search and clustering of dense vectors](https://github.com/facebookresearch/faiss)
  - [ElasticSearch](https://github.com/elastic/elasticsearch)
  - [SPTAG: A library for fast approximate nearest neighbor search](https://github.com/microsoft/SPTAG)
  - [OpenMatch: An Open-Source Package for Information Retrieval](https://github.com/thunlp/OpenMatch)

