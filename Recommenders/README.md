### TensorFlow Recommenders Examples 

0. [Quick Start](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/Recommenders/0-Recommenders-Quickstart.ipynb)
1. [Retrieval Model](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/Recommenders/1-Retrieval.ipynb)
2. [Ranking Model](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/Recommenders/2-Ranking.ipynb)
3. [Feature Preprocessisng](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/Recommenders/3-Feature-preprocessing.ipynb)
4. [Context Features](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/Recommenders/4-Leveraging-context-features.ipynb)
5. [Deep Retrieval Model](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/Recommenders/5-Building-deep-retrieval-models.ipynb)
6.  [Multi-task Recommenders](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/Recommenders/6-Multi-task-recommenders.ipynb)
7.  [Deep & Cross Network](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/Recommenders/7-Deep%26Cross-Network%20(DCN).ipynb)
8.  [Efficient Serving](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/Recommenders/8-Efficient-serving.ipynb)
9.  [Listwise Ranking](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/Recommenders/9-Listwise-ranking.ipynb)

### Cold Start Model Deep Dive

Using sampled data set, run the code to build the similar models

1. [Retrieval Model](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/Recommenders/1-Retrieval-Cold_Start_20210909.ipynb)
2. [Ranking Model](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/Recommenders/2-Ranking-Cold_Start_20210909.ipynb)
3. [Feature Preprocessisng](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/Recommenders/3-Feature-preprocessing-Cold_Start_20210910.ipynb)
4. [Context Features](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/Recommenders/4-Leveraging-context-features-Cold_start_20210911.ipynb)
5. [Deep Retrieval Model](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/Recommenders/5-Building-deep-retrieval-models-Cold_Start_20210912.ipynb)

### Candidate Generation Training

1. [Debug Deep Retrieval Model 20210913](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/6-candidate-generation-training_20210913.ipynb)
2. [SageMaker Deep Retrieval Model 20210914](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/6-candidate-generation-training_20210914.ipynb)
3. [SageMaker Deep Retrieval Model 20210915](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/6-candidate-generation-training_20210915.ipynb)

### Fine Tuning

1. [Broadcaster embedding dim 20210927](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/7-Investigate_embedding_20210927.ipynb)
1. [Early stopping 20211207](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/12-Investigating_early_stop_20211207.ipynb)
1. [Dropout 20211124](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/11-Investigate_dropout_20211124.ipynb)
1. [SageMaker Dropout 20211207](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/11-Investigate_dropout_20211207_epoches20_sagemaker.ipynb)

### Feature Importance

1. [Permutation Feature Importance 20211022](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/8-Investigate-Feature-Importance-20211022.ipynb)
2. [SHAP Feature Importance 20211025](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/8-Investigate-Feature-Importance-20211025.ipynb)

### Feature Engineering

1. Viewer age bucket
     * [User age bucket analysis 20210924](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/7-Investigate_viewer_age_20210924.ipynb)
     * [SageMaker User age bucket analysis 20210930](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/7-Investigate_viewer_age_20210930.ipynb)
     
2. Viewer latitude and longitude 
     * [Viewer latitude and longitude analysis 20211001](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/7-Investigate_viewer_lat_log_20211001.ipynb)
     * [Viewer latitude and longitude analysis 20211012](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/7-Investigate_viewer_lat_long_cluster_20211012.ipynb)
     * [SageMaker Viewer latitude and longitude analysis 20211006](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/7-Investigate_viewer_lat_long_cluster_20211006.ipynb)
     
3. Broadcaster trending
     * [Broadcaster trending 20211028](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/7-Investigate_broadcaster_trending_20211028.ipynb)
     * [Broadcaster trending 20211103](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/7-Investigate_broadcaster_trending_20211103.ipynb)

4. Implicit feedback from search broadcast
     * [Implicit feedback search broadcast  20220106](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/13-Lightgbm_negative_signal_20220106.ipynb)
     * [Implicit feedback search broadcast  20220110](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/13-Lightgbm_negative_signal_20220110.ipynb)
     
### Ranking Models

1. Estimating total watch time using lightgbm
     * [Negative signal duration lightgbm  regression 20211108](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/9-Lightgbm_negative_signal_20211108.ipynb)
     * [Negative signal duration lightgbm regression 20211110](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/9-Lightgbm_negative_signal_20211110.ipynb)

2. Estimating watch time Longer than cutoffs using lightgbm
     * [Optimization metrics lightgbm classification  20211115](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/10-Lightgbm_optimization_metrics_20211115.ipynb)
     * [Optimization metrics lightgbm classification  20211118](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/10-Optimization_metrics_20211118.ipynb)

3. Estimating watch time using tfrs
     * [Ranking model  20220112](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/14-Ranking_20220112.ipynb)
     * [Ranking model  20220114](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/14-Ranking_20220114.ipynb)

### Multi-task Models

1. Comparison of retrieval, ranking, and joint models
     * [Multi-task model 20220103](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/14-Ranking_20220103_comparison.ipynb)
     
     * [Multi-task model 20220118](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/14-Ranking_20220118_comparison.ipynb)

2. Batch Normalization
     * [BatchNormalization 20220128](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/15-BatchNormalization_20220128.ipynb)

### A/B Testing

1. Hyperparameter comparison
     * [Experiments 20220222](https://github.meetmecorp.com/lhuang/Deep-recommender-sandbox/blob/master/notebooks/16-Experiments_20220222.ipynb)
