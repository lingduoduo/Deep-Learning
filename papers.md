## WSDM 2022

**跨域推荐**

RecGURU: Adversarial Learning of Generalized User Representations for Cross-Domain Recommendation

https://arxiv.org/pdf/2111.10093.pdf

Personalized Transfer of User Preferences for Cross-domain Recommendation

https://arxiv.org/pdf/2110.11154.pdf

Multi-Sparse-Domain Collaborative Recommendation via Enhanced Comprehensive Aspect Preference Learning

**序列推荐**

Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation

https://arxiv.org/pdf/2110.05730.pdf

S-Walk: Accurate and Scalable Session-based Recommendation with Random Walks

Heterogeneous Global Graph Neural Networks for Personalized Session-based Recommendation

https://arxiv.org/pdf/2107.03813.pdf

Learning Multi-granularity Consecutive User Intent Unit for Session-based Recommendation

**点击率预估**

CAN: Feature Co-Action Network for Click-Through Rate Prediction

Triangle Graph Interest Network for Click-through Rate Prediction

Modeling Users’ Contextualized Page-wise Feedback for Click-Through Rate Prediction in E-commerce Search

**去偏推荐**

It Is Different When Items Are Older: Debiasing Recommendations When Selection Bias and User Preferences are Dynamic

https://arxiv.org/pdf/2111.12481.pdf

Fighting Mainstream Bias in Recommender Systems via Local Fine Tuning

http://people.tamu.edu/~zhuziwei/pubs/Ziwei_WSDM_2022.pdf

Towards Unbiased and Robust Causal Ranking for Recommender Systems

**路径推荐**

PLdFe-RR:Personalized Long-distance Fuel-efficient Route Recommendation Based On Historical Trajectory

**联邦推荐**

PipAttack: Poisoning Federated Recommender Systems for Manipulating Item Promotion

https://arxiv.org/pdf/2110.10926.pdf

**基于图结构的推荐**

Joint Learning of E-commerce Search and Recommendation with A Unified Graph Neural Network

Profiling the Design Space for Graph Neural Networks based Collaborative Filtering

http://www.shichuan.org/doc/125.pdf

Graph Logic Reasoning for Recommendation and Link Prediction

Modeling Scale-free Graphs with Hyperbolic Geometry for Knowledge-aware Recommendation

https://arxiv.org/pdf/2108.06468.pdf

**公平性推荐**

Toward Pareto Efficient Fairness-Utility Trade-off in Recommendation through Reinforcement Learning

Enumerating Fair Packages for Group Recommendations

https://arxiv.org/pdf/2105.14423.pdf

**基于对比学习的推荐**

Contrastive Meta Learning with Behavior Multiplicity for Recommendation

C2-CRS: Coarse-to-Fine Contrastive Learning for Conversational Recommender System

**基于元学习的推荐**

Long Short-Term Temporal Meta-learning in Online Recommendation

https://arxiv.org/pdf/2105.03686.pdf

**基于对抗学习的推荐**

A Peep into the Future: Adversarial Future Encoding in Recommendation

**基于强化学习的推荐**

Reinforcement Learning over Sentiment-Augmented Knowledge Graphs towards Accurate and Explainable Recommendation

A Cooperative-Competitive Multi-Agent Framework for Auto-bidding in Online Advertising

https://arxiv.org/pdf/2106.06224.pdf

Choosing the Best of All Worlds: Accurate, Diverse, and Novel Recommendations through Multi-Objective Reinforcement Learning

https://arxiv.org/pdf/2110.15097.pdf

**关于数据集**

On Sampling Collaborative Filtering Datasets

The Datasets Dilemma: How Much Do We Really Know About Recommendation Datasets?

其他

VAE++: Variational AutoEncoder for Heterogeneous One-Class Collaborative Filtering

Sequential Modeling with Multiple Attributes for Watchlist Recommendation in E-Commerce

https://arxiv.org/pdf/2110.11072.pdf

Show Me the Whole World: Towards Entire Item Space Exploration for Interactive Personalized Recommendations

https://arxiv.org/pdf/2110.09905.pdf

Supervised Advantage Actor-Critic for Recommender Systems

https://arxiv.org/pdf/2111.03474.pdf

**官网接收论文列表地址：**

https://www.wsdm-conference.org/2022/accepted-papers/


## KDD 2021
1. Categorize by usage

主要挑选了一些笔者比较感兴趣的方向，并整理了对应的文章名称。读者可以大致读一下文章名，判断是否和自己的研究方向或工作方向一致，从中选择感兴趣的文章进行精读。

1.1 Recommendations

1.1.1 Sampling

涉及到采样、负样本等。

- Google: Bootstrapping for Batch Active Sampling
- Google: Bootstrapping Recommendations at Chrome Web Store
- Alibaba：Real Negatives Matter: Continuous Training with Real Negatives for Delayed Feedback Modeling

1.1.2 Representation Learning

- Google: Learning to Embed Categorical Features without Embedding Tables for Recommendation
- 华为：An Embedding Learning Framework for Numerical Features in CTR Prediction
- 腾讯：Learning Reliable User Representations from Volatile and Sparse Data to Accurately Predict Customer Lifetime Value
- 阿里：Representation Learning for Predicting Customer Orders

1.1.3 Cross-domain recommendation

- 阿里：Debiasing Learning based Cross-domain Recommendation
- 腾讯：Adversarial Feature Translation for Multi-domain Recommendation

1.1.4 Debiasing learning

- 阿里：Contrastive Learning for Debiased Candidate Generation in Large-Scale Recommender Systems
- 阿里：Debiasing Learning based Cross-domain Recommendation

1.1.5 Graph Neural Network

- 华为：Dual Graph enhanced Embedding Neural Network for CTR Prediction
- 美团：Signed Graph Neural Network with Latent Groups
- 阿里：DMBGN: Deep Multi-Behavior Graph Networks for Voucher Redemption Rate Prediction
- 百度：MugRep: A Multi-Task Hierarchical Graph Representation Learning Framework for Real Estate Appraisal

1.1.6 Multi-task learning

- Google：Understanding and Improving Fairness-Accuracy Trade-offs in Multi-Task Learning
- 美团：Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning for Customer Acquisition
- 百度：MugRep: A Multi-Task Hierarchical Graph Representation Learning Framework for Real Estate Appraisal

1.1.7 Micro-Video recommendations

- 阿里：SEMI: A Sequential Multi-Modal Information Transfer Network for E-Commerce Micro-Video Recommendations

1.1.8 Knowledge Graph generation

- Microsoft：Reinforced Anchor Knowledge Graph Generation for News Recommendation Reasoning

1.1.9 Recommender Infrastruture

- Facebook：Training Recommender Systems at Scale: Communication-Efficient Model and Data Parallelism
- Facebook：Hierarchical Training: Scaling Deep Recommendation Models on Large CPU Clusters
- 阿里，FleetRec: Large-Scale Recommendation Inference on Hybrid GPU-FPGA Clusters
- 腾讯，Large-Scale Network Embedding in Apache Spark
- Microsoft，On Post-Selection Inference in A/B Testing

1.2 Search

1.2.1 Embedding 

- 阿里：Embedding-based Product Retrieval in Taobao Search

1.2.2 Query understanding

- Facebook：Que2Search: Fast and Accurate Query and Document Understanding for Search at Facebook

1.2.3 Knowledge Graph

- 阿里巴巴：AliCG: Fine-grained and Evolvable Conceptual Graph Construction for Semantic Search at Alibaba
- 阿里巴巴：AliCoCo2: Commonsense Knowledge Extraction, Representation and Application in E-commerce

1.2.4 Pretraining

- 百度：Pretrained Language Models for Web-scale Retrieval in Baidu Search
- 微软：Domain-Specific Pretraining for Vertical Search: Case Study on Biomedical Literature

1.2.5 Query rewriting and auto-completion

- 微软：Diversity driven Query Rewriting in Search Advertising
- 百度：Meta-Learned Spatial-Temporal POI Auto-Completion for the Search Engine at Baidu Maps

1.2.6 Graph Attention 

- 百度：HGAMN: Heterogeneous Graph Attention Matching Network for Multilingual POI Retrieval at Baidu Maps

1.2.7 Multitask

- Google: Mondegreen: A Post-Processing Solution to Speech Recognition Error Correction for Voice Search Queries
- Facebook：VisRel: Media Search at Scale

1.2.8 Feature interaction

- 阿里：FIVES: Feature Interaction Via Edge Search for Large-Scale Tabular Data

1.2.9 Serice

- 百度：Norm Adjusted Proximity Graph for Fast Inner Product Retrieval
- 百度：JIZHI: A Fast and Cost-Effective Model-As-A-Service System for Web-Scale Online Inference at Baidu

1.3 Ads

这一块文章不是很多，就不细分了。

- Google: Clustering for Private Interest-based Advertising
- 阿里：A Unified Solution to Constrained Bidding in Online Display Advertising
- 阿里：Exploration in Online Advertising Systems with Deep Uncertainty-Aware Learning
- 阿里：Neural Auction: End-to-End Learning of Auction Mechanisms for E-Commerce Advertising
- 阿里：We Know What You Want: An Advertising Strategy Recommender System for Online Advertising

1.4 NLP

1.4.1 Transformer

- 微软：NAS-BERT: Task-Agnostic and Adaptive-Size BERT Compression with Neural Architecture Search
- 阿里：M6: Multi-Modality-to-Multi-Modality Multitask Mega-transformer for Unified Pretraining
- 微软：TUTA: Tree-based Transformers for Generally Structured Table Pre-training

1.4.2 Named Entity Recognition

- 微软：Reinforced Iterative Knowledge Distillation for Cross-Lingual Named Entity Recognition

1.4.3 Multi-label learning

- 微软：Generalized Zero-Shot Extreme Multi-label Learning
- 微软：Zero-shot Multi-lingual Interrogative Question Generation for "People Also Ask" at Bing

1.4.4 Attractive

- 微软：Reinforcing Pretrained Models for Generating Attractive Text Advertisements

1.4.5 User Intent classification

- 阿里：MeLL: Large-scale Extensible User Intent Classification for Dialogue Systems with Meta Lifelong Learning

1.4.6 Multi-Modality

- 阿里：M6: Multi-Modality-to-Multi-Modality Multitask Mega-transformer for Unified Pretraining

2 Categorize by Company

2.1 Google

- Learning to Embed Categorical Features without Embedding Tables for Recommendation
- NewsEmbed: Modeling News through Pre-trained Document Representations
- Understanding and Improving Fairness-Accuracy Trade-offs in Multi-Task Learning
- Bootstrapping for Batch Active Sampling
- Bootstrapping Recommendations at Chrome Web Store
- Clustering for Private Interest-based Advertising
- Dynamic Language Models for Continuously Evolving Content
- Mondegreen: A Post-Processing Solution to Speech Recognition Error Correction for Voice Search Queries
- On Training Sample Memorization: Lessons from Benchmarking Generative Modeling with a Large-scale Competition

2.2 Facebook

- Training Recommender Systems at Scale: Communication-Efficient Model and Data Parallelism
- Preference Amplification in Recommender Systems
- Hierarchical Training: Scaling Deep Recommendation Models on Large CPU Clusters
- Network Experimentation at Scale
- Que2Search: Fast and Accurate Query and Document Understanding for Search at Facebook
- VisRel: Media Search at Scale
- Balancing Consistency and Disparity in Network Alignment

2.3 Microsoft

- Generalized Zero-Shot Extreme Multi-label Learning
- Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport
- NAS-BERT: Task-Agnostic and Adaptive-Size BERT Compression with Neural Architecture Search
- Reinforced Anchor Knowledge Graph Generation for News Recommendation Reasoning
- Table2Charts: Recommending Charts by Learning Shared Table Representations
- TabularNet: A Neural Network Architecture for Understanding Semantic Structures of Tabular Data
- TUTA: Tree-based Transformers for Generally Structured Table Pre-training
- Contextual Bandit Applications in a Customer Support Bot
- Diversity driven Query Rewriting in Search Advertising
- Domain-Specific Pretraining for Vertical Search: Case Study on Biomedical Literature
- On Post-Selection Inference in A/B Testing
- Reinforced Iterative Knowledge Distillation for Cross-Lingual Named Entity Recognition
- Reinforcing Pretrained Models for Generating Attractive Text Advertisements
- Zero-shot Multi-lingual Interrogative Question Generation for "People Also Ask" at Bing

2.4 阿里

- A Unified Solution to Constrained Bidding in Online Display Advertising
- AliCG: Fine-grained and Evolvable Conceptual Graph Construction for Semantic Search at Alibaba
- AliCoCo2: Commonsense Knowledge Extraction, Representation and Application in E-commerce
- Contrastive Learning for Debiased Candidate Generation in Large-Scale Recommender Systems
- Debiasing Learning based Cross-domain Recommendation
- Device-Cloud Collaborative Learning for Recommendation
- Deep Inclusion Relation-aware Network for User Response Prediction at Fliggy
- DMBGN: Deep Multi-Behavior Graph Networks for Voucher Redemption Rate Prediction
- Dual Attentive Sequential Learning for Cross-Domain Click-Through Rate Prediction
- Embedding-based Product Retrieval in Taobao Search
- Exploration in Online Advertising Systems with Deep Uncertainty-Aware Learning
- FIVES: Feature Interaction Via Edge Search for Large-Scale Tabular Data
- FleetRec: Large-Scale Recommendation Inference on Hybrid GPU-FPGA Clusters
- Intention-aware Heterogeneous Graph Attention Networks for Fraud Transactions Detection
- Live-Streaming Fraud Detection: A Heterogeneous Graph Neural Network Approach
- M6: Multi-Modality-to-Multi-Modality Multitask Mega-transformer for Unified Pretraining
- Markdowns in E-Commerce Fresh Retail: A Counterfactual Prediction and Multi-Period Optimization Approach
- MeLL: Large-scale Extensible User Intent Classification for Dialogue Systems with Meta Lifelong Learning
- Multi-Agent Cooperative Bidding Games for Multi-Objective Optimization in e-Commercial Sponsored Search
- Neural Auction: End-to-End Learning of Auction Mechanisms for E-Commerce Advertising
- Real Negatives Matter: Continuous Training with Real Negatives for Delayed Feedback Modeling
- Representation Learning for Predicting Customer Orders
- SEMI: A Sequential Multi-Modal Information Transfer Network for E-Commerce Micro-Video Recommendations
- We Know What You Want: An Advertising Strategy Recommender System for Online Advertising

2.5 百度

- Norm Adjusted Proximity Graph for Fast Inner Product Retrieval
- Curriculum Meta-Learning for Next POI Recommendation
- Pretrained Language Models for Web-scale Retrieval in Baidu Search
- HGAMN: Heterogeneous Graph Attention Matching Network for Multilingual POI Retrieval at Baidu Maps
- JIZHI: A Fast and Cost-Effective Model-As-A-Service System for Web-Scale Online Inference at Baidu
- Meta-Learned Spatial-Temporal POI Auto-Completion for the Search Engine at Baidu Maps
- MugRep: A Multi-Task Hierarchical Graph Representation Learning Framework for Real Estate Appraisal
- SSML: Self-Supervised Meta-Learner for En Route Travel Time Estimation at Baidu Maps
- Talent Demand Forecasting with Attentive Neural Sequential Model

2.6 腾讯

- Why Attentions May Not Be Interpretable?
- Adversarial Feature Translation for Multi-domain Recommendation
- Large-Scale Network Embedding in Apache Spark
- Learn to Expand Audience via Meta Hybrid Experts and Critics
- Learning Reliable User Representations from Volatile and Sparse Data to Accurately Predict Customer Lifetime Value

 2.7 美团

- Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning for Customer Acquisition
- User Consumption Intention Prediction in Meituan
- Signed Graph Neural Network with Latent Groups
- A Deep Learning Method for Route and Time Prediction in Food Delivery Service

2.8 华为

- An Embedding Learning Framework for Numerical Features in CTR Prediction
- Dual Graph enhanced Embedding Neural Network for CTR Prediction
- Discrete-time Temporal Network Embedding via Implicit Hierarchical Learning
- Retrieval & Interaction Machine for Tabular Data Prediction
- A Multi-Graph Attributed Reinforcement Learning Based Optimization Algorithm for Large-scale Hybrid Flow Shop Scheduling Problem
