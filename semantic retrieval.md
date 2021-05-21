<div align='center' ><font size='70'>Semantic Retrieval</font></div>

[TOC]

## Survey Paper

- [Semantic Models for the First-stage Retrieval: A Comprehensive Review.](https://arxiv.org/pdf/2103.04831.pdf) 






## Classical Term-based Retrieval

- 【VSM】A vector space model for automatic indexing（1975）
- 【TFIDF】Developments in automatic text retrieval（1991）
- 【TFIDF】Term-weighting approaches in automatic text retrieval.（1988）
- 【Binary Independence Model】Relevance weighting of search terms（1976）
- 【概率模型】Relevance weighting of search terms（1977）、Probabilistic models of  indexing and searching（1981）、The uniﬁed probabilistic model for ir（1983）、Efﬁcient probabilistic inference for text retrieval（1991）、Some simple effective approximations to the 2-poisson model for probabilistic weighted retrieval（1994）、The Probability Ranking Principle in IR（1997，提出概率排序准则）、The uniﬁed model revisited（2003）、A new uniﬁed probabilistic model（2004）
- 【Tree Dependence Model】A theoretical basis for the use of co-occurrence data in information retrieval
- 【BM25】The probabilistic relevance framework: BM25 and beyond（2010）
- 【DFR, Divergence from Randomness】Probabilistic models of information retrieval based on measuring the divergence from randomness（2002，随即距离模型）
- 【语言模型/QL】A language modeling approach to information retrieval（1998）
- 【语言模型】Statistical language models for information retrieval a critical review（zhai,2007）
- 【语言模型】Hypergeometric language model and Zipf-like scoring function for web document similarity retrieval（2010）



## Early Methods for Semantic Retrieval

#### Query Expansion

- Global methods expand or reformulate query words by analyzing the word co-occurrences from the corpus being searched or using hand-crafted thesaurus
- Global analysis often relies on external resources or document collections
- Word-word associations in document retrieval systems.（1969）
- Concept based query expansion（1993）
- 【RM3】Relevance based language models（2001）
- Model-based feedback in the language modeling approach to information retrieval.（2001）
- Query expansion using local and global document analysis.
- 【伪相关反馈】Query expansion using lexical-semantic relations.（1994）
- 【伪相关反馈】Selecting good expansion terms for pseudo-relevance feedback.（2008）
- 【Rocchio伪相关反馈】Relevance Feedback in Information Retrieval（1971）
- Search Engines: Information Retrieval in Practice.（2009）
- A comparative study of methods for estimating query language models with pseudo feedback.（2009）
- Pseudo-relevance feedback based on matrix factorization（2016）
- Reducing the risk of query expansion via robust constrained optimization（2009，基于伪相关反馈的查询扩展容易遭受query drift的问题）

#### Term Dependency Model

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

#### Topic Model

- 【GVSM】Generalized vector spaces model in information retrieval.（1985）
- 【LSA/LSI】Indexing by latent semantic analysis.（1990）
- 【NMF】Algorithms for non-negative matrix factorization.（2001）
- 【RLSI】Regularized latent semantic indexing（2011）、Group matrix factorization for scalable topic modeling.（2012）
- 【PLSA】Probabilistic latent semantic indexing（1999）
- 【LDA】Latent dirichlet allocation（2003）
- Supervised topic models.（2007）
- Investigating task performance of probabilistic topic models: an empirical study of PLSA and LDA.（2010）
- Corpus structure, language models, and ad hoc information retrieval.（2004，topic 模型用于平滑项）
- Regularizing ad hoc retrieval scores（2005，topic 模型用于平滑项）
- Lda-based document models for ad-hoc retrieval.（2006，LDA for IR 和 LDA用于LM的平滑）
- A comparative study of utilizing topic models for information retrieval.（2009）
- Latent semantic indexing (lsi) fails for trec collections（2011）

#### Translation Model

- Information retrieval as statistical translation.（1999）
- Estimation of statistical translation models based on mutual information for ad hoc information retrieval.（2010）
- Axiomatic analysis of translation language model for information retrieval.（2012）
- Language model for information retrieval（2002）





## Neural Methods for Semantic Retrieval
#### Sparse Retrieval Methods

- 【DeepTR】Learning to reweight terms with distributed representations（2015）
- 【NTLM】Integrating and evaluating neural word embeddings in information retrieval（2015）
- 【TVD】Learning Term Discrimination（SIGIR2020）
- 【DeepCT】Context-Aware Sentence/Passage Term Importance Estimation for First Stage Retrieval（arxiv2019）
- 【DeepCT】Context-Aware Term Weighting For First-Stage Passage Retrieval（SIGIR2020）
- 【DeepCT】Efficiency Implications of Term Weighting for Passage Retrieval（SIGIR2020）
- 【HDCT】Context-Aware Document Term Weighting for Ad-Hoc Search（WWW2020）
- 【GAR】Generation-augmented retrieval for open-domain question answering（2020，查询扩展）
- 【Doc2Query】Document Expansion by query Prediction.（2019）
- 【DocTTTTTQuery】From doc2query to docTTTTTquery（2019）
- SparTerm: Learning Term-based Sparse Representation for Fast Text Retrieval（2020）
- SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking（SIGIR2021）
- Learning Passage Impacts for Inverted Indexes（SIGIR2021）
- Semantic hashing（2009）
- 【SNRM】From Neural Re-Ranking to Neural Ranking: Learning a Sparse Representation for Inverted Indexing（CIKM2018）
- UHD-BERT: Bucketed Ultra-High Dimensional Sparse Representations for Full Ranking（2021）

#### Dense Retrieval Methods

- 【FV】Aggregating continuous word embeddings for information retrieval（2013）
- Short Text Similarity with Word Embeddings（CIKM2015）
- 【DESM】A Dual Embedding Space Model for Document Ranking（2016）
- Efficient natural language response suggestion for smart reply（2017）
- 【PIQA】Phrase indexed question answering: A new challenge for scalable document comprehension.（EMNLP2018）
- End-to-end retrieval in continuous space（2018）
- 【MUPPET】Multi-Hop Paragraph Retrieval for Open-Domain Question Answering（ACL2019）
- 【Poly-encoders】Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring（ICLR2020）
- 【DPR】Dense Passage Retrieval for Open-Domain Question Answering（2020）
- 【RepBERT】RepBERT: CONTEXTUALIZED TEXT EMBEDDINGS FOR FIRST-STAGE RETRIEVAL（2020）
- 【DC-BERT】DC-BERT: Decoupling Question and Document for Efficient Contextual Encoding（2020）
- 【ColBERT】ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT（SIGIR2020）
- 【ME-BERT】Sparse, Dense, and Attentional Representations for Text Retrieval（2020）
- 【ANCE】Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval
- 【RocketQA】An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering
- 【TCT-ColBERT】Distilling Dense Representations for Ranking using Tightly-Coupled Teachers
- Distilling Knowledge for Fast Retrieval-based Chat-bots（2020）
- Learning to Retrieve：how to train a dense retrieval model effectively and efficiently
- Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling（SIGIR2021）
- Optimizing Dense Retrieval Model Training with Hard Negatives（SIGIR2021）
- Joint Learning of Deep Retrieval Model and Product Quantization based Embedding Index（SIGIR2021）
- Improving Document Representations by Generating Pseudo Query Embeddings for Dense Retrieval（2021）
- COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List（2021）
- 加速
  - Incorporating query term independence assumption for efficient retrieval and ranking using deep neural networks（2019）
  - Efficient Interaction-based Neural Ranking with Locality Sensitive Hashing（WWW2019）
  - 【PreTTR】Efficient Document Re-Ranking for Transformers by Precomputing Term Representations（SIGIR2020）
  - 【DeFormer】DeFormer: Decomposing Pre-trained Transformers for Faster Question Answering（ACL2020）
  - SPARTA: Efﬁcient Open-Domain Question Answering via Sparse Transformer Matching Retrieval
  - Conformer-kernel with query term independence for document retrieval（2020）
- 预训练
  - 【ORQA】Latent Retrieval for Weakly Supervised Open Domain Question Answering（ACL2019
  - 【REALM】 Retrieval-Augmented Language Model Pre-Training（2020）
  - Pre-training Tasks for Embedding-based Large-scale Retrieval（2020）
  - Embedding-based Zero-shot Retrieval through Query Generation（query生成）
  - Zero-shot neural retrieval via domain-targeted synthetic query generation.（2020，query生成）

#### Hybrid of Sparse-dense Retrieval Methods

- Monolingual and cross-lingual information retrieval models based on (bilingual) word embeddings.（2015）
-  【GLM】 Word embedding based generalized language model for information retrieval.（2015）
- 【BOW-CNN】Learning Hybrid Representations to Retrieve Semantically Equivalent Questions（2015）
- Representing documents and queries as sets of word embedded vectors for information retrieval.（2016）
-  【DESM】A Dual Embedding Space Model for Document Ranking（2016）
=======
- 【GLM】 Word embedding based generalized language model for information retrieval.（2015）
- 【BOW-CNN】Learning Hybrid Representations to Retrieve Semantically Equivalent Questions（2015）
- Representing documents and queries as sets of word embedded vectors for information retrieval.（2016）
- 【DESM】A Dual Embedding Space Model for Document Ranking（2016）
- 【DenSPI】Real-Time Open-Domain Question Answering with Dense-Sparse Phrase Index（ACL2019）
- 【SPARC】Contextualized Sparse Representations for Real-Time Open-Domain Question Answering（ACL2020）
- 【EPIC】Expansion via Prediction of Importance with Contextualization（SIGIR2020）
- 【Hybrid】Sparse, Dense, and Attentional Representations for Text Retrieval（2020）
- 【CLEAR】Complementing lexical retrieval with semantic residual embedding.（2020，线性加权结合BM25和bert-based model）
- Leveraging Semantic and Lexical Matching to Improve the Recall of Document Retrieval Systems: A Hybrid Approach（2020）






## Other Resources
#### Other Tasks
- Embedding-based Retrieval in Facebook Search（KDD2020）
- K-way D-dimensional Discrete Codes for Compact Embedding Representations
- Deep Retrieval: An End-to-End Learnable Structure Model for Large-Scale Recommendations
- From Semantic Retrieval to Pairwise Ranking: Applying Deep Learning in E-commerce Search（SIGIR2019）
- 【DPSR】Towards Personalized and Semantic Retrieval: An End-to-End Solution for E-commerce Search via Embedding Learning（SIGIR2020）
- MOBIUS: Towards the Next Generation of Query-Ad Matching in Baidu’s Sponsored Search（百度，凤巢）
- 【DIN】Deep interest network for click-through rate prediction
- 【MIND】Multi-Interest Network with Dynamic Routing for Recommendation at Tmall（2019，天猫）
- 【DMIN】Deep Multi-Interest Network for Click-through Rate Prediction（CIKM2020）
-  Report on the First HIPstIR Workshop on the Future of Information Retrieval（2019，workshop）
-  Let’s measure run time! Extending the IR replicability infrastructure to include performance aspects.（SIGIR2019）
-  Off the Beaten Path: Let’s Replace Term-Based Retrieval with k-NN Search（2016CIKM）

#### Datasets
- 【TREC CAR】TREC Complex Answer Retrieval Overview
- 【MS MARCO】A human-generated machine reading comprehension datase

#### Index
- 基于树的方法
  - 【KD树】Multidimensional binary search trees used for associative searching（1975）
- 哈希方法
  - 【LSH】Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality（1998）
- 矢量量化方法
  - 【PQ】Product quantization for nearest neighbor search（2010）
- 基于图的方法
  - 【NSW】Navigation in a small world
  - 【HNSW】Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs
- Toolkits
  - [Faiss: a library for efficient similarity search and clustering of dense vectors](https://github.com/facebookresearch/faiss)
