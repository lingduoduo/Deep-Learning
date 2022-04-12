TensorFlow is used in almost every google application for machine learning. You are using TensorFlow indirectly while using applications like Google Voice Search or Google Photos. 
TensorFlow works like a computational library for writing new algorithms, that involves a large number of Tensor operations. It is widely used for deep learning applications as well since neuron numbers can be easily expressed as computational graphs. 
So, they can be implemented using TensorFlow as a series of operations on Tensors. 
Solve the key features of TensorFlow are as following. 
TensorFlow efficiently works with mathematical expressions involving multi-dimensional arrays. 
TensorFlow completely supports the deep neural networks and machine learning concepts. 
TensorFlow supports GPU CPU computing where the same code can be executed on both architectures. 
Parallelism is one of the top advantages of TensorFlow, meaning that you can execute your computation on graph parallely. 
You are going to have a control over the execution, and you can schedule different tasks on different processors like GPU CPU and etc. 
But, what is the Tensor. Tensor is the core concept for TensorFlow. 
TensorFlow uses a Tensor data structure to represent all data. 
Only Tensors are passed between the operations in TensorFlow. 
You can think of a TensorFlow Tensor as an n dimensional array or at least. 
A Tensor has a rank a shape and a static type. 
So, a Tensor can be represented as a multi-dimensional array of numbers.


A well separated cluster is a set of objects in which each object is closer to every other object in the same cluster compared to the other clusters. Normally, a threshold is defined for the comparison. 

A prototype based cluster consists of a set of objects in reach each object is closer to the prototype that defines that at a specific cluster. An example of the prototype based cluster would be centroid or mean for continuous data, and for categorical data, the most representative member of the cluster. 

Another type of the clusters is the graph based cluster. If the data is represented as a graph where the nodes are the objects and the edges show the connections, every connected part is a cluster such that its elements are all related. They are not related to the other clusters. This clustering is not suitable for noisy data. An example of this cluster type would be continuity based clustering. 

Density-based clusters are another type of clusters, which is defined as following. A dense region of objects that is surrounded by a region of low density. This definition is often employed when the clusters are irregular or intertwine, and noise and outliers exist in the data. Shared property or conceptual clusters include sets of objects that share some specific properties. This definition encompasses all other definitions of clusters. 

In order to solve different clustering problems, there are various clustering algorithms such as K-Means clustering, hierarchical clustering, t-SNE clustering, and DBSCAN or DB Scan clustering. 


TFX: Real World Machine Learning in Production

What are pipelines and componentts?
A TFX component has three main parts: a driver, an executor, and a publisher.
A driver, coordinating job execution, which inspects the state of world and decides what work nees to be done, coordinating job executiona dn feeding metadata to the executor.
A executor, performing the work, which inserts your code and do your customization.
A publisher, updating ML metadata, which takes the results of your executor and updates the metadata store.

Orchestrating a TFX pipeline
TFX implements a task- and data-aware pipeline architecture.

Why should you store metadata
Store information/artifacts which the models were trained.
Keep execution records ML pipeline run frequently.
Include lineage or provenance of the data objects as they flow through the pipeline.

Metadata-powered functionality
Remember that it’s not just for today’s model or today’s results. You’re likely also interested in understanding how your data and results change over time as you take in new data and retrain your model. You often want to compare to model runs that you ran yesterday, or last week, to understand why your results got better or worse. Production solutions aren’t one-time things, they live for as long as you need them, and that can be months or years.

Read in data
ExampleGen
StatisticsGen
SchemaGen
ExampleValidator

Feature Engineering
Transform will then output a TensorFlow graph with those constants and ops. 
That graph is hermetic, so it contains all of the information you need to apply those transformations, and will form the input stage for your model. 

Training Models -
Trainer - transform graphs and data
SchemaGen - schema
Evaluator - savedModel
ModelValidator - pusher
