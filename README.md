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
  - [Document Expansion](#document-expansion)
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

- [Semantic Models for the First-stage Retrieval: A Comprehensive Review](https://arxiv.org/pdf/2103.04831.pdf) (Cai et.al., 2021, Arxiv)



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

### Document Expansion

- to do

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
  - [Learning to reweight terms with distributed representations](https://dl.acm.org/doi/pdf/10.1145/2766462.2767700)（Zheng et.al., 2015, SIGIR, **DeepTR**）
  - [Integrating and evaluating neural word embeddings in information retrieval](https://dl.acm.org/doi/pdf/10.1145/2838931.2838936)（Zuccon et.al., 2015, ADCS, **NTLM**）
  - [Learning Term Discrimination](https://dl.acm.org/doi/pdf/10.1145/3397271.3401211)（Frej et.al, 2020, SIGIR, **TVD**）
  - [Context-Aware Sentence/Passage Term Importance Estimation for First Stage Retrieval](https://arxiv.org/pdf/1910.10687.pdf)（Dai et.al., 2019, arXiv, **DeepCT**）
  - [Context-Aware Term Weighting For First-Stage Passage Retrieval](https://dl.acm.org/doi/pdf/10.1145/3397271.3401204)（Dai et.al., 2020, SIGIR, **DeepCT**）
  - [Efficiency Implications of Term Weighting for Passage Retrieval](https://dl.acm.org/doi/pdf/10.1145/3397271.3401263)（Mackenzie et.al., 2020, SIGIR, **DeepCT**）
  - [Context-Aware Document Term Weighting for Ad-Hoc Search](https://dl.acm.org/doi/pdf/10.1145/3366423.3380258)（Dai et.al., 2020, WWW, **HDCT**）
  - [A Few Brief Notes on DeepImpact, COIL, and a Conceptual Framework for Information Retrieval Techniques](https://arxiv.org/pdf/2106.14807.pdf)（Lin et.al., 2021, arXiv, **uniCOIL**）<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />
  
- **Expansion**
  - [Generation-augmented retrieval for open-domain question answering](https://aclanthology.org/2021.acl-long.316.pdf)（Mao et.al., 2020, ACL, **GAR**, query expansion）
  - [Document Expansion by query Prediction](https://arxiv.org/pdf/1904.08375.pdf)（Nogueira et.al., 2019, arXiv, **Doc2Query**）
  - [From doc2query to docTTTTTquery](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf)（Nogueira et.al., 2019, arXiv, **DocTTTTTQuery**）
  - [A Unified Pretraining Framework for Passage Ranking and Expansion](https://ojs.aaai.org/index.php/AAAI/article/view/16584)（Yan et.al., 2021, AAAI, **UED**）
- **Expansion + Term Re-weighting**
  - [SparTerm: Learning Term-based Sparse Representation for Fast Text Retrieval](https://arxiv.org/pdf/2010.00768.pdf)（Bai et.al., 2020, arXiv, **SparTerm**）
  - [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://dl.acm.org/doi/pdf/10.1145/3404835.3463098)（Formal et.al., 2021, SIGIR, **SPLADE**）
  - [Learning Passage Impacts for Inverted Indexes](https://dl.acm.org/doi/pdf/10.1145/3404835.3463030)（Mallia et.al., 2021, SIGIR, **DeepImapct**）
- **Sparse Representation**
  - [Semantic hashing](https://reader.elsevier.com/reader/sd/pii/S0888613X08001813?token=DC834E84448C134BC820C604CD63095AD2736144D81932A2E4EF5080206926924B21983C3254D232BB02587EDBC6F1AE&originRegion=us-east-1&originCreation=20210806113324)（Salakhutdinov et.al., 2009）
  - [From Neural Re-Ranking to Neural Ranking: Learning a Sparse Representation for Inverted Indexing](https://dl.acm.org/doi/pdf/10.1145/3269206.3271800)（Zamani et.al., 2018, CIKM, **SNRM**）
  - [UHD-BERT: Bucketed Ultra-High Dimensional Sparse Representations for Full Ranking](https://arxiv.org/pdf/2104.07198.pdf)（Jang et.al., 2021, arXiv, **UHD-BERT**）
  - [Efﬁcient Passage Retrieval with Hashing for Open-domain Question Answering](https://arxiv.org/pdf/2106.00882.pdf)（Yamada et.al., 2021, ACL, **BPR**）<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />

### Dense Retrieval Methods

- **Word-embedding-based**
  - Aggregating continuous word embeddings for information retrieval（2013, **FV**）
  - Short Text Similarity with Word Embeddings（2015, CIKM）
  - A Dual Embedding Space Model for Document Ranking（2016, **DESM**）
  - Efficient natural language response suggestion for smart reply（2017）
  - End-to-end retrieval in continuous space（2018）
- Phrase indexed question answering: A new challenge for scalable document comprehension（2018, EMNLP, **PIQA**）
- CoRT: Complementary Rankings from Transformers（2019, ICTIR, **CoRT**）
- Dense Passage Retrieval for Open-Domain Question Answering（2020, **DPR**）
- RepBERT: CONTEXTUALIZED TEXT EMBEDDINGS FOR FIRST-STAGE RETRIEVAL（2020, arXiv, **RepBERT**）
- DC-BERT: Decoupling Question and Document for Efficient Contextual Encoding（2020, **DC-BERT**）
- Neural Retrieval for Question Answering with Cross-Attention Supervised Data Augmentation
- Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval（**ANCE**）
- An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering（**RocketQA**）
- Learning to Retrieve：how to train a dense retrieval model effectively and efficiently（**LTRe**）
- Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling（2021, SIGIR）
- Optimizing Dense Retrieval Model Training with Hard Negatives（2021, SIGIR, **STAR/ADORE**）
- Learning Dense Representations of Phrases at Scale（2021, **DensePhrases**）
- More Robust Dense Retrieval with Contrastive Dual Learning（**DANCE**）<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />
- PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval（2021, ACL, **PAIR**）<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />
- Relevance-guided Supervision for OpenQA with ColBERT（2021, **ColBERT-QA**）<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />
- End-to-End Training of Multi-Document Reader and Retriever for Open-Domain Question Answering（2021）<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />
- Distilling Knowledge from Reader to Retriever for Question Answering<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />
- **Knowledge Distillation**
  - Distilling Dense Representations for Ranking using Tightly-Coupled Teachers（**TCT-ColBERT**）
  - Distilling Knowledge for Fast Retrieval-based Chat-bots（2020）
  - Improving Bi-encoder Document Ranking Models with Two Rankers and Multi-teacher Distillation
  - Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation（2021，Margin-MSE loss）
- **Multi-vector Representation** 
  - Multi-Hop Paragraph Retrieval for Open-Domain Question Answering（2019, ACL, **MUPPET**）
  - Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring（2020, ICLR, **Poly-encoders**）
  - Sparse, Dense, and Attentional Representations for Text Retrieval（2020, **ME-BERT**）
  - ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT（2020, SIGIR, **ColBERT**）
  - COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List（2021, **COIL**）
  - Improving Document Representations by Generating Pseudo Query Embeddings for Dense Retrieval（2021, ACL）
- **Accelerate Interaction-based Models**
  - Incorporating query term independence assumption for efficient retrieval and ranking using deep neural networks（2019）
  - Efficient Interaction-based Neural Ranking with Locality Sensitive Hashing（2019, WWW）
  - Efficient Document Re-Ranking for Transformers by Precomputing Term Representations（2020, SIGIR, **PreTTR**）
  - DeFormer: Decomposing Pre-trained Transformers for Faster Question Answering（2020, ACL, **DeFormer**）
  - SPARTA: Efﬁcient Open-Domain Question Answering via Sparse Transformer Matching Retrieval（**SPARTA**）
  - Conformer-kernel with query term independence for document retrieval（2020）
- **Pre-training**
  - Latent Retrieval for Weakly Supervised Open Domain Question Answering（2019, ACL, **ORQA**）
  - Retrieval-Augmented Language Model Pre-Training（2020, **REALM**）
  - Pre-training Tasks for Embedding-based Large-scale Retrieval（2020）
  - Embedding-based Zero-shot Retrieval through Query Generation（query generation）
  - Zero-shot neural retrieval via domain-targeted synthetic query generation（2020，query generation）
  - Towards Robust Neural Retrieval Models with Synthetic Pre-Training（query generation）
  - Pre-trained Language Model for Web-scale Retrieval in Baidu Search<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />
- **Joint Learning with Index**
  - Joint Learning of Deep Retrieval Model and Product Quantization based Embedding Index（2021, SIGIR）
  - Jointly Optimizing Query Encoder and Product Quantization to Improve Retrieval Performance（2021）<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />

### Hybrid of Sparse-dense Retrieval Methods

- Monolingual and cross-lingual information retrieval models based on (bilingual) word embeddings（2015）
- Word embedding based generalized language model for information retrieval（2015, **GLM**）
- Learning Hybrid Representations to Retrieve Semantically Equivalent Questions（2015, **BOW-CNN**）
- Word embedding based generalized language model for information retrieval（2015）
- Representing documents and queries as sets of word embedded vectors for information retrieval（2016）
- A Dual Embedding Space Model for Document Ranking（2016, **DESM_mixture**）
- Real-Time Open-Domain Question Answering with Dense-Sparse Phrase Index (2019, ACL, **DenSPI**)
- Contextualized Sparse Representations for Real-Time Open-Domain Question Answering（2020, ACL, **SPARC**）
- Expansion via Prediction of Importance with Contextualization（2020, SIGIR, **EPIC**）
- Sparse, Dense, and Attentional Representations for Text Retrieval（2020, **Hybrid**）
- Complementing lexical retrieval with semantic residual embedding（2020, **CLEAR**）
- Leveraging Semantic and Lexical Matching to Improve the Recall of Document Retrieval Systems: A Hybrid Approach（2020）
- Contextualized Offline Relevance Weighting for Efficient and Effective Neural Retrieval（2021, SIGIR）<img src="./images/new.gif" alt="image-20191107150755839" style="zoom:150%;" />




## Other Resources
### Other Tasks
- Embedding-based Retrieval in Facebook Search（KDD2020）
- K-way D-dimensional Discrete Codes for Compact Embedding Representations
- Deep Retrieval: An End-to-End Learnable Structure Model for Large-Scale Recommendations
- From Semantic Retrieval to Pairwise Ranking: Applying Deep Learning in E-commerce Search（SIGIR2019，Jingdong）
- Towards Personalized and Semantic Retrieval: An End-to-End Solution for E-commerce Search via Embedding Learning（2020, SIGIR, **DPSR**, Jingdong）
- MOBIUS: Towards the Next Generation of Query-Ad Matching in Baidu’s Sponsored Search（Baidu）
- Deep interest network for click-through rate prediction（**DIN**）
- Multi-Interest Network with Dynamic Routing for Recommendation at Tmall（2019, **MIND**, Tmall）
- Deep Multi-Interest Network for Click-through Rate Prediction（2020, CIKM, **DMIN**）
- Few-Shot Conversational Dense Retrieval（2021, SIGIR, dialogue）
- Report on the First HIPstIR Workshop on the Future of Information Retrieval（2019, workshop）
- Let’s measure run time! Extending the IR replicability infrastructure to include performance aspects（2019, SIGIR）
- Off the Beaten Path: Let’s Replace Term-Based Retrieval with k-NN Search（2016, CIKM）
- Binary Neural Network Hashing for Image Retrieval（2021, SIGIR）

### Datasets
- 【TREC CAR】TREC Complex Answer Retrieval Overview
- 【MS MARCO】A human-generated machine reading comprehension datase
- 【TREC DL】

### Index
- **Tree-based**
  - 【KD tree】Multidimensional binary search trees used for associative searching（1975）
- **Hashing-based**
  - 【LSH】Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality（1998）
  - Asymmetric LSH for sublinear time maximum inner product search（2014）
- **Quantization-based**
  - Product quantization for nearest neighbor search（2010, **PQ**）
  - Optimized product quantization（**OPQ**）
  - Additive quantization for extreme vector compression（2014）
  - Composite quantization for approximate nearest neighbor search（2014）
  - Tree quantization for large-scale similarity search and classification（2015）
- **Graph-based**
  - Scalable k-NN graph construction for visual descriptors（2012）
  - Navigation in a small world（NSW）
  - Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs（HNSW）
- Toolkits
  - [Faiss: a library for efficient similarity search and clustering of dense vectors](https://github.com/facebookresearch/faiss)
  - [ElasticSearch](https://github.com/elastic/elasticsearch)
  - [SPTAG: A library for fast approximate nearest neighbor search](https://github.com/microsoft/SPTAG)
  - [OpenMatch: An Open-Source Package for Information Retrieval](https://github.com/thunlp/OpenMatch)

