# Semantic Retrieval

## 传统方法

- [ ] VSM：A vector space model for automatic indexing（1975）

- [ ] TFIDF：Developments in automatic text retrieval（1991）

- [ ] TFIDF：Term-weighting approaches in automatic text retrieval.（1988）

- [ ] Binary Independence Model：Relevance weighting of search terms（1976）

- [ ] 概率模型：Relevance weighting of search terms（1977）、Probabilistic models of  indexing and searching（1981）、The uniﬁed probabilistic model for ir（1983）、Efﬁcient probabilistic inference for text retrieval（1991）、Some simple effective approximations to the 2-poisson model for probabilistic weighted retrieval（1994）、The Probability Ranking Principle in IR（1997，提出概率排序准则）、The uniﬁed model revisited（2003）、A new uniﬁed probabilistic model（2004）

- [ ] Tree Dependence Model：A theoretical basis for the use of co-occurrence data in information retrieval

- [ ] BM25：The probabilistic relevance framework: BM25 and beyond（2010）

- [ ] DFR（Divergence from Randomness）：Probabilistic models of information retrieval based on measuring the divergence from randomness（2002，随即距离模型）

- [ ] 语言模型/QL：A language modeling approach to information retrieval（1998）

- [ ] 语言模型：Statistical language models for information retrieval a critical review（zhai,2007）

- [ ] 语言模型：Hypergeometric language model and Zipf-like scoring function for web document similarity retrieval（2010）

## 查询扩展

- [ ] Global methods expand or reformulate query words by analyzing the word co-occurrences from the corpus being searched or using hand-crafted thesaurus
- [ ] Global analysis often relies on external resources or document collections
- [ ] Word-word associations in document retrieval systems.（1969）
- [ ] Concept based query expansion（1993）
- [ ] RM3：Relevance based language models（2001）
- [ ] Model-based feedback in the language modeling approach to information retrieval.（2001）
- [ ] Query expansion using local and global document analysis.

- [ ] 伪相关反馈：Query expansion using lexical-semantic relations.（1994）
- [ ] 伪相关反馈：Selecting good expansion terms for pseudo-relevance feedback.（2008）
- [ ] Relevance Feedback in Information Retrieval（1971，Rocchio伪相关反馈）
- [ ] Search Engines: Information Retrieval in Practice.（2009）
- [ ] A comparative study of methods for estimating query language models with pseudo feedback.（2009）
- [ ] Pseudo-relevance feedback based on matrix factorization（2016）
- [ ] Reducing the risk of query expansion via robust constrained optimization（2009，基于伪相关反馈的查询扩展容易遭受query drift的问题）

## Term Dependency Model

- [ ] VSM + term dependency
  - Experiments in automatic phrase indexing for document retrieval: A comparison of syntactic and non-syntactic methods（1987）
  - An analysis of statistical and syntactic phrases（1997）
  - Term-weighting approaches in automatic text retrieval（1988）
  - A probabilistic model of information retrieval: development and comparative experiments.（2000）
- [ ] BM25 + term dependency
  - Relevance ranking using kernels.（2010）
- [ ] LM + term dependency 
  - A general language model for information retrieval.（1999）
  - Biterm language models for document retrieval.（2002）
  - Capturing term dependencies using a language model based on sentence trees.（2002）
  - Dependence language model for information retrieval（2004）
- [ ] A generative theory of relevance（2004）
- [ ] SDM：A Markov random ﬁeld model for term dependencies（2005）

## Topic Model

- [ ] GVSM：Generalized vector spaces model in information retrieval.（1985）

- [ ] LSA/LSI：Indexing by latent semantic analysis.（1990）

- [ ] NMF：Algorithms for non-negative matrix factorization.（2001）

- [ ] RLSI：Regularized latent semantic indexing（2011）、Group matrix factorization for scalable topic modeling.（2012）

- [ ] PLSA：Probabilistic latent semantic indexing（1999）

- [ ] LDA： Latent dirichlet allocation（2003）

- [ ] Supervised topic models.（2007）

- [ ] Investigating task performance of probabilistic topic models: an empirical study of PLSA and LDA.（2010）

- [ ] Corpus structure, language models, and ad hoc information retrieval.（2004，topic 模型用于平滑项）

- [ ] Regularizing ad hoc retrieval scores（2005，topic 模型用于平滑项）

- [ ] Lda-based document models for ad-hoc retrieval.（2006，LDA for IR 和 LDA用于LM的平滑）

- [ ] A comparative study of utilizing topic models for information retrieval.（2009）


- [ ] Latent semantic indexing (lsi) fails for trec collections（2011）




- [ ] 指明快速推断对于IR的重要性，提出两层无向图模型：Exponential family harmoniums with an application to information retrieval.（2005）

- [ ] 无向图模型：The rate adapting poisson (RAP) model for information retrieval and object recognition.（2006）

- [ ] 无向图模型：Mining associated text and images with dual-wing harmoniums.（2005）

## Translation Model

- [ ] Information retrieval as statistical translation.（1999l）

- [ ] Estimation of statistical translation models based on mutual information for ad hoc information retrieval.（2010）

- [ ] Axiomatic analysis of translation language model for information retrieval.（2012）

- [ ] Language model for information retrieval（2002）


## Using Word Embedding

- [x] 【Fisher】Aggregating continuous word embeddings for information retrieval（2013）

## Neural-based Model

### Improving Term-based Model

##### Neural-based Weighting Schemes

- 使用静态word embedding

- [x] 【NTLM】Integrating and evaluating neural word embeddings in information retrieval（2015）
- [x]  【DeepTR】Learning to reweight terms with distributed representations（2015）
- [x] Learning Term Discrimination（SIGIR2020）
- [ ] Query expansion with locally-trained word embeddings（2016）
- [ ] Using word embeddings for automatic query expansion（2016）
- [ ] Embedding-based query language models（2016）

- 使用contextual embedding

- [x] 【DeepCT】Context-Aware Sentence/Passage Term Importance Estimation for First Stage Retrieval（arxiv2019）
- [x] 【DeepCT】Context-Aware Term Weighting For First-Stage Passage Retrieval（SIGIR2020）
- [x] 【HDCT】Context-Aware Document Term Weighting for Ad-Hoc Search（WWW2020）
- [x] 【DeepCT】Efficiency Implications of Term Weighting for Passage Retrieval（SIGIR2020）

- 用于document expansion

- [x] 【Doc2Query】Document Expansion by query Prediction.（2019，使用seq2seq生成多个query，扩充原始文档）
- [x] 【DocTTTTTQuery】From doc2query to docTTTTTquery.（2019，doc2query的进一步工作，使用T5生成query）

##### Hybrid of Term-based and Neural-based Models

- [x] Monolingual and cross-lingual information retrieval models based on (bilingual) word embeddings.（2015）
- [x] 【GLM】 Word embedding based generalized language model for information retrieval.（2015）
- [x] 【DESM】A Dual Embedding Space Model for Document Ranking（2016）
- [ ] Improving Document Ranking with Dual Word Embeddings
- [x] Representing documents and queries as sets of word embedded vectors for information retrieval.（2016）

- [x] 【CLEAR】Complementing lexical retrieval with semantic residual embedding.（2020，线性加权结合BM25和bert-based model）

### Designing Representation-focused Model

##### Dense

- word embedding
  
  - [x] End-to-end retrieval in continuous space（2018）
  
- 基于表示的模型
  - [x] DSSM
  - [x] ArcI
  - [x] QA-LSTM
  
- 专门的基于表示的模型

  - [x] 【DC-BERT】DC-BERT: Decoupling Question and Document for Efficient Contextual Encoding（2020）
  - [x] 【ColBERT】ColBERT: Eicient and Eective Passage Search via Contextualized Late Interaction over BERT（SIGIR2020）
  - [x] 【PIQA】Phrase indexed question answering: A new challenge for scalable document comprehension.（EMNLP2018）
  - [x] 【MUPPET】Multi-Hop Paragraph Retrieval for Open-Domain Question Answering（ACL2019）
  - [x] Sparse, Dense, and Attentional Representations for Text Retrieval（2020）
  - [x] 【Poly-encoders】Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring（ICLR2020）
  - [x] Efficient natural language response suggestion for smart reply.（2017）
  - [x] 【DPR】Dense Passage Retrieval for Open-Domain Question Answering（2020）
  - [x] RepBERT: CONTEXTUALIZED TEXT EMBEDDINGS FOR FIRST-STAGE RETRIEVAL（2020）
  
- 预训练模型

  - [x] 【ORQA】Latent Retrieval for Weakly Supervised Open Domain Question Answering（ACL2019）

  - [x] 【REALM】: Retrieval-Augmented Language Model Pre-Training（2020）
  - [x] Pre-training Tasks for Embedding-based Large-scale Retrieval（2020）
  - [x] Embedding-based Zero-shot Retrieval through Query Generation（query生成，预训练）

##### Sparse

- [x] Semantic hashing
- [x] 【SNRM】From Neural Re-Ranking to Neural Ranking: Learning a Sparse Representation for Inverted Indexing（CIKM2018）

##### Dense-sparse Hybrid

- [x] 【DenSPI】Real-Time Open-Domain Question Answering with Dense-Sparse Phrase Index（ACL2019）

- [x] 【SPARC】Contextualized Sparse Representations for Real-Time Open-Domain Question Answering（ACL2020）
- [x] Sparse, Dense, and Attentional Representations for Text Retrieval（2020）

### Modify Interaction-focused Model

- [x] Incorporating query term independence assumption for efficient retrieval and ranking using deep neural networks（2019）
- [x] Efficient Interaction-based Neural Ranking with Locality Sensitive Hashing（WWW2019）
- [x] 【PreTTR】Efficient Document Re-Ranking for Transformers by Precomputing Term Representations（SIGIR2020）
- [x] 【DeFormer】DeFormer: Decomposing Pre-trained Transformers for Faster Question Answering（ACL2020）



## 综述

- [x] Neural Matching and Importance Learning in Information Retrieval（Dai博士论文）




## ANN等快速搜索算法

- [ ] Ann-benchmarks: A benchmarking tool for approximate nearest neighbor algorithms.（2017）
- [ ] Fast Item Ranking under Neural Network based Measures（WSDM2020，基于图的近似近邻检索）
- [ ] Billion-scale similarity search with GPUs（2017，faiss，hps://github.com/facebookresearch/faiss）
- [ ] Quantization based fast inner product search（2016）
- [x] Near-optimal hashing algorithms for approximate nearest neighbor in high dimensions.（2006，局部敏感哈希）
- [ ] 局部敏感哈希：Locality-sensitive hashing scheme based on p-stable distributions.（2004）
- [ ] ！！距离敏感哈希：Distance-sensitive hashing（2018）



## 数据集

- [ ] 【TREC CAR】TREC Complex Answer Retrieval Overview

- [ ] 【MS Marco】A human-generated machine reading comprehension datase



## 其他

- [x] Embedding-based Retrieval in Facebook Search（KDD2020）
- [x] Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval
- [x] Debiased Contrastive Learning
- [x] moco:Momentum contrast for unsupervised visual representation learning; Improved baselines with momentum contrastive learning
- [x] K-way D-dimensional Discrete Codes for Compact Embedding Representations
- [x] Deep Retrieval: An End-to-End Learnable Structure Model for Large-Scale Recommendations
- [x] Zero-shot neural retrieval via domain-targeted synthetic query generation.（2020）
- [x] Report on the First HIPstIR Workshop on the Future of Information Retrieval（2019，workshop）
- [x] Let’s measure run time! Extending the IR replicability infrastructure to include performance aspects.（SIGIR2019）
- [ ] SSI：Supervised semantic indexing

- [x] Off the Beaten Path: Let’s Replace Term-Based Retrieval with k-NN Search（2016CIKM）

##### 电商、推荐、广告

- [x] From Semantic Retrieval to Pairwise Ranking: Applying Deep Learning in E-commerce Search（SIGIR2019）
- [x] 【DPSR】Towards Personalized and Semantic Retrieval: An End-to-End Solution for E-commerce Search via Embedding Learning（SIGIR2020）
- [ ] Learning a Joint Search and Recommendation Model from User-Item Interactions（WSDM2020）
- [ ] Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations.（WWW2020）
- [ ] Sampling-bias-corrected neural modeling for large corpus item recommendations.（2019）
- [ ] ！！Deep Neural Networks for YouTube Recommendations.（2016）
- [ ] Multi-Interest Network with Dynamic Routing for Recommendation at Tmall.（CIKM2019）
- [ ] Learning Tree-Based Deep Model for Recommender Systems.（KDD2018）
- [ ] ！！Learning latent vector spaces for product search.（CIKM2016，用word embedding，entity embedding）
- [x] MOBIUS: Towards the Next Generation of Query-Ad Matching in Baidu’s Sponsored Search
- [ ] ！！Deep interest network for click-through rate prediction
- [x] Multi-Interest Network with Dynamic Routing for Recommendation at Tmall（2019，天猫）




