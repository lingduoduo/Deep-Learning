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

User - User

- Who to follow
- When an user follows another, are they interested in all the contnet from that user?

User - Item

-  Home timeline
- What is right training objective: CTR, fav, reply, retweet, DAU/MAU

Item - Item (similarity and diversity)

- Tweet - tweet similarity
- Trend - Trends similarity (use case is folding) more recent = less engagement

What is an item at Twitter referred to as content

- tweets
- events/moments
- trends
- live video / broadcasts

User intests will vary over time

- long term interests
- short term interests
- geo-dependant interests/happendings
- privacy and control is extremely important
- striking the right balance betwen the is key, geo-dependant flavor

Metrics

- Coverage: given the large and even changing catalog of content/items, gettign high coverage with low latency is oxymoronic

- Diversity

- Utility: if glancing the items may have enough utility to inform the user without clicking on it. how to factor into an objective function for training

- Adaptability: given ever changing world of content, how fast can the system adapt? how does system adapt to users with low engagement(new/light) vs heavy users?

- Scalability: For billions of users across the globe, with unique habits and preferences

- User Preference: Handlign explicti interests and learned itnerests together in one ranking framework


## Tensorflow

Tensorflow教程(tensorflow.org)

https://www.tensorflow.org/tutorials/

Tensorflow入门--CPU vs GPU (medium.com/@erikhallstrm)

https://medium.com/@erikhallstrm/hello-world-tensorflow-649b15aed18c

Tensorflow入门(metaflow.fr)

https://blog.metaflow.fr/tensorflow-a-primer-4b3fa0978be3

Tensorflow实现RNNs (wildml.com)

http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/

Tensorflow实现文本分类CNN模型(wildml.com)

http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

如何用Tensorflow做文本摘要(surmenok.com)

http://pavel.surmenok.com/2016/10/15/how-to-run-text-summarization-with-tensorflow/

## PyTorch

Pytorch教程(pytorch.org)

http://pytorch.org/tutorials/

Pytorch快手入门 (gaurav.im)

http://blog.gaurav.im/2017/04/24/a-gentle-intro-to-pytorch/

利用Pytorch深度学习教程(iamtrask.github.io)

https://iamtrask.github.io/2017/01/15/pytorch-tutorial/

Pytorch实战(github.com/jcjohnson)

https://github.com/jcjohnson/pytorch-examples

PyTorch 教程(github.com/MorvanZhou)

https://github.com/MorvanZhou/PyTorch-Tutorial

深度学习研究人员看的PyTorch教程(github.com/yunjey)

https://github.com/yunjey/pytorch-tutorial

## Deep Learning

果壳里的深度学习(nikhilbuduma.com)

http://nikhilbuduma.com/2014/12/29/deep-learning-in-a-nutshell/

深度学习教程 (Quoc V. Le)

http://ai.stanford.edu/~quocle/tutorial1.pdf

深度学习，什么鬼？(machinelearningmastery.com)

http://machinelearningmastery.com/what-is-deep-learning/

什么是人工智能，机器学习，深度学习之间的区别？ (nvidia.com)

https://blogs.nvidia.com/blog/2016/07/29/whats-difference-artificial-intelligence-machine-learning-deep-learning-ai/

Practical Deep Learning for Coders — Part 1 http://course.fast.ai/

deep learning book http://www.deeplearningbook.org/

Google Colaboratory https://colab.research.google.com/

Big Picture of Calculus https://ocw.mit.edu/resources/res-18-005-highlights-of-calculus-spring-2010/highlights_of_calculus/big-picture-of-calculus/

Linear Algebra https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/

Matrix Calculus for Deep Learning http://parrt.cs.usfca.edu/doc/matrix-calculus/index.html

Deep Learning Specialisation https://www.coursera.org/specializations/deep-learning

Cutting Edge Deep Learning for Coders http://course.fast.ai/part2.html

CS231n http://cs231n.stanford.edu/

CS224d http://cs224d.stanford.edu/

11、优化算法与降维算法

数据降维的七招炼金术(knime.org)

https://www.knime.org/blog/seven-techniques-for-data-dimensionality-reduction

主成分分析(Stanford CS229)

http://cs229.stanford.edu/notes/cs229-notes10.pdf 

Dropout: 改进神经网络的一个简单方法(Hinton @ NIPS 2012)

http://videolectures.net/site/normal_dl/tag=741100/nips2012_hinton_networks_01.pdf

如何溜你们家的深度神经网络？(rishy.github.io)

http://rishy.github.io/ml/2017/01/05/how-to-train-your-dnn/

12、长短期记忆(LSTM) 

老司机带你简易入门长短期神经网络(machinelearningmastery.com)

http://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/

理解LSTM网络(colah.github.io)

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

漫谈LSTM模型(echen.me)

http://blog.echen.me/2017/05/30/exploring-lstms/

小学生看完这教程都可以用Python实现一个LSTM-RNN (iamtrask.github.io)

http://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/

13、卷积神经网络（CNNs）

卷积网络入门(neuralnetworksanddeeplearning.com)

http://neuralnetworksanddeeplearning.com/chap6.html#introducing_convolutional_networks

深度学习与卷积神经网络模型(medium.com/@ageitgey)

https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721

拆解卷积网络模型(colah.github.io)

http://colah.github.io/posts/2014-07-Conv-Nets-Modular/

理解卷积网络(colah.github.io)

http://colah.github.io/posts/2014-07-Understanding-Convolutions/

14、递归神经网络(RNNs)

递归神经网络教程 (wildml.com)

http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/

注意力模型与增强型递归神经网络(distill.pub)

http://distill.pub/2016/augmented-rnns/

这么不科学的递归神经网络模型(karpathy.github.io)

http://karpathy.github.io/2015/05/21/rnn-effectiveness/

深入递归神经网络模型(nikhilbuduma.com)

http://nikhilbuduma.com/2015/01/11/a-deep-dive-into-recurrent-neural-networks/

15、强化学习

给小白看的强化学习及其实现指南 (analyticsvidhya.com)

https://www.analyticsvidhya.com/blog/2017/01/introduction-to-reinforcement-learning-implementation/

强化学习教程(mst.edu)

https://web.mst.edu/~gosavia/tutorial.pdf

强化学习，你学了么？(wildml.com)

http://www.wildml.com/2016/10/learning-reinforcement-learning/

深度强化学习：开挂玩Pong (karpathy.github.io)

http://karpathy.github.io/2016/05/31/rl/

16、对抗式生成网络模型(GANs) 

什么是对抗式生成网络模型？(nvidia.com)

https://blogs.nvidia.com/blog/2017/05/17/generative-adversarial-network/

用对抗式生成网络创造8个像素的艺术(medium.com/@ageitgey)

https://medium.com/@ageitgey/abusing-generative-adversarial-networks-to-make-8-bit-pixel-art-e45d9b96cee7

对抗式生成网络入门（TensorFlow）(aylien.com)

http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/

《对抗式生成网络》（小学一年级~上册）(oreilly.com)

https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners

17、多任务学习

深度神经网络中的多任务学习概述(sebastianruder.com)

http://sebastianruder.com/multi-task/index.html


### 数学

1、机器学习中的数学 

机器学习中的数学 (ucsc.edu)

https://people.ucsc.edu/~praman1/static/pub/math-for-ml.pdf

机器学习数学基础(UMIACS CMSC422)

http://www.umiacs.umd.edu/~hal/courses/2013S_ML/math4ml.pdf

2、线性代数

线性代数简明指南(betterexplained.com)

https://betterexplained.com/articles/linear-algebra-guide/

码农眼中矩阵乘法 (betterexplained.com)

https://betterexplained.com/articles/matrix-multiplication/

理解叉乘运算(betterexplained.com)

https://betterexplained.com/articles/cross-product/

理解点乘运算(betterexplained.com)

https://betterexplained.com/articles/vector-calculus-understanding-the-dot-product/

机器学习中的线性代数(U. of Buffalo CSE574)

http://www.cedar.buffalo.edu/~srihari/CSE574/Chap1/LinearAlgebra.pdf

深度学习的线代小抄(medium.com)

https://medium.com/towards-data-science/linear-algebra-cheat-sheet-for-deep-learning-cd67aba4526c

复习线性代数与课后阅读材料(Stanford CS229)

http://cs229.stanford.edu/section/cs229-linalg.pdf

3、概率论

贝叶斯理论 (betterexplained.com)

https://betterexplained.com/articles/understanding-bayes-theorem-with-ratios/

理解贝叶斯概率理论(Stanford CS229)

http://cs229.stanford.edu/section/cs229-prob.pdf

复习机器学习中的概率论(Stanford CS229)

https://see.stanford.edu/materials/aimlcs229/cs229-prob.pdf

概率论(U. of Buffalo CSE574)

http://www.cedar.buffalo.edu/~srihari/CSE574/Chap1/Probability-Theory.pdf

机器学习中的概率论(U. of Toronto CSC411)

http://www.cs.toronto.edu/~urtasun/courses/CSC411_Fall16/tutorial1.pdf

4、计算方法（Calculus）

如何理解导数：求导法则，指数和算法(betterexplained.com)

https://betterexplained.com/articles/how-to-understand-derivatives-the-quotient-rule-exponents-and-logarithms/

如何理解导数，乘法，幂指数，链式法(betterexplained.com)

https://betterexplained.com/articles/derivatives-product-power-chain/

向量计算，理解梯度(betterexplained.com)

https://betterexplained.com/articles/vector-calculus-understanding-the-gradient/

微分计算(Stanford CS224n)

http://web.stanford.edu/class/cs224n/lecture_notes/cs224n-2017-review-differential-calculus.pdf

计算方法概论(readthedocs.io)

http://ml-cheatsheet.readthedocs.io/en/latest/calculus.html

1、机器学习就是这么好玩！(medium.com/@ageitgey)

机器学习速成课程（Berkeley的ML）

Part I：https://ml.berkeley.edu/blog/2016/11/06/tutorial-1/

Part II：https://ml.berkeley.edu/blog/2016/12/24/tutorial-2/

Part III：https://ml.berkeley.edu/blog/2017/02/04/tutorial-3/

机器学习入门与应用：实例图解(toptal.com)

https://www.toptal.com/machine-learning/machine-learning-theory-an-introductory-primer

机器学习的简易指南 (monkeylearn.com)

https://monkeylearn.com/blog/a-gentle-guide-to-machine-learning/

如何选择机器学习算法？(sas.com)

https://blogs.sas.com/content/subconsciousmusings/2017/04/12/machine-learning-algorithm-use/

2、Activation and Loss Functions 激活函数与损失函数

sigmoid 神经元 (neuralnetworksanddeeplearning.com)

http://neuralnetworksanddeeplearning.com/chap1.html#sigmoid_neurons

激活函数在神经网络中有什么作用？(quora.com)

https://www.quora.com/What-is-the-role-of-the-activation-function-in-a-neural-network

神经网络的激活函数大全及其优劣 (stats.stackexchange.com)

https://stats.stackexchange.com/questions/115258/comprehensive-list-of-activation-functions-in-neural-networks-with-pros-cons

激活函数及其分类比较(medium.com)

https://medium.com/towards-data-science/activation-functions-and-its-types-which-is-better-a9a5310cc8f

理解对数损失 (exegetic.biz)

http://www.exegetic.biz/blog/2015/12/making-sense-logarithmic-loss/

损失函数(Stanford CS231n)

http://cs231n.github.io/neural-networks-2/#losses

损失函数L1 与L2 比较(rishy.github.io)

http://rishy.github.io/ml/2015/07/28/l1-vs-l2-loss/

交叉熵损失函数(neuralnetworksanddeeplearning.com)

http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function

3、偏差（Bias）

神经网络中的偏差的作用(stackoverflow.com)

https://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks/2499936#2499936

神经网络中的偏差节点(makeyourownneuralnetwork.blogspot.com)

http://makeyourownneuralnetwork.blogspot.com/2016/06/bias-nodes-in-neural-networks.html

什么是人工神经网络中的偏差 (quora.com)

https://www.quora.com/What-is-bias-in-artificial-neural-network

4、感知器（Perceptron）

感知器模型(neuralnetworksanddeeplearning.com)

http://neuralnetworksanddeeplearning.com/chap1.html#perceptrons

感知器(natureofcode.com)

http://natureofcode.com/book/chapter-10-neural-networks/#chapter10_figure3 

一层的神经网络（感知器模型）(dcu.ie)

http://computing.dcu.ie/~humphrys/Notes/Neural/single.neural.html

从感知器模型到深度网络(toptal.com)

https://www.toptal.com/machi                         ne-learning/an-introduction-to-deep-learning-from-perceptrons-to-deep-networks

5、回归算法

线性回归分析简介(duke.edu)

http://people.duke.edu/~rnau/regintro.htm

线性回归 (ufldl.stanford.edu)

http://ufldl.stanford.edu/tutorial/supervised/LinearRegression/

线性回归 (readthedocs.io)

http://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html

逻辑斯特回归 (readthedocs.io)

http://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html 

机器学习之简单线性回归教程(machinelearningmastery.com)

http://machinelearningmastery.com/simple-linear-regression-tutorial-for-machine-learning/

机器学习之逻辑斯特回归教程(machinelearningmastery.com)

http://machinelearningmastery.com/logistic-regression-tutorial-for-machine-learning/

softmax 回归(ufldl.stanford.edu)

http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/

6、梯度下降

基于梯度下降的学习 (neuralnetworksanddeeplearning.com)

http://neuralnetworksanddeeplearning.com/chap1.html#learning_with_gradient_descent

梯度下降(iamtrask.github.io)

http://iamtrask.github.io/2015/07/27/python-network-part2/

如何理解梯度下降算法？(kdnuggets.com)

http://www.kdnuggets.com/2017/04/simple-understand-gradient-descent-algorithm.html

梯度下降优化算法概览(sebastianruder.com)

http://sebastianruder.com/optimizing-gradient-descent/

优化算法：随机梯度下降算法 (Stanford CS231n)

http://cs231n.github.io/optimization-1/

7、生成学习

生成学习算法 (Stanford CS229)

http://cs229.stanford.edu/notes/cs229-notes2.pdf

贝叶斯分类算法之实例解析(monkeylearn.com)

https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/

8、支持向量机

支持向量机（SVM）入门(monkeylearn.com)

https://monkeylearn.com/blog/introduction-to-support-vector-machines-svm/

支持向量机(Stanford CS229)

http://cs229.stanford.edu/notes/cs229-notes3.pdf

线性分类：支持向量机，Softmax (Stanford 231n)

http://cs231n.github.io/linear-classify/

9、后向传播算法（Backpropagation）

后向传播算法必知(medium.com/@karpathy)

https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b

来，给我图解一下神经网络后向传播算法？(github.com/rasbt)

https://github.com/rasbt/python-machine-learning-book/blob/master/faq/visual-backpropagation.md

后向传播算法是如何运行的？(neuralnetworksanddeeplearning.com)

http://neuralnetworksanddeeplearning.com/chap2.html

沿时后向传播算法与梯度消失(wildml.com)

http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/

简易入门沿时后向传播算法(machinelearningmastery.com)

http://machinelearningmastery.com/gentle-introduction-backpropagation-time/

奔跑吧，后向传播算法！(Stanford CS231n)

http://cs231n.github.io/optimization-2/

