**Tensorflow**
- TensorFlow is used in almost every google application for machine learning. You are using TensorFlow indirectly while using applications like Google Voice Search or Google Photos. 
- TensorFlow works like a computational library for writing new algorithms, that involves a large number of Tensor operations. It is widely used for deep learning applications as well since neuron numbers can be easily expressed as computational graphs. 
So, they can be implemented using TensorFlow as a series of operations on Tensors. 
Solve the key features of TensorFlow are as following. 
- TensorFlow efficiently works with mathematical expressions involving multi-dimensional arrays. 
- TensorFlow completely supports the deep neural networks and machine learning concepts. 
- TensorFlow supports GPU CPU computing where the same code can be executed on both architectures. 
Parallelism is one of the top advantages of TensorFlow, meaning that you can execute your computation on graph parallely. 
You are going to have a control over the execution, and you can schedule different tasks on different processors like GPU CPU and etc. 
But, what is the Tensor. Tensor is the core concept for TensorFlow. 
- TensorFlow uses a Tensor data structure to represent all data. 
Only Tensors are passed between the operations in TensorFlow. 
You can think of a TensorFlow Tensor as an n dimensional array or at least. 
- A Tensor has a rank a shape and a static type. 
So, a Tensor can be represented as a multi-dimensional array of numbers.

**Tensors**
- Tensors is n-dimensional array, container for holding data, a mathematical representation for the magnigudes of a colleciton of features.
- Tensors use rank, shape, and dtype to describe the data they hold.
    - Rank of a tensor is the number of indices required to uniquesly select each element of the tensor
    - Rank 0 Tensor: Scalar
    - Rank 1 Tensor: Vector (numpy narray)
    - Rank 2 Tensor: Matrix
    - Rank 3 Tensor: Tensor
**Flow**
- Flow is movement and change, Directed Acyclic Graph
    - Directed: you can traverse the graph in a predefinced sequence of steps.
    - Acyclic: no looping, data doesn't go through the same node more than once.
    - Graph: nodes and edges will represent operations and data, respectively
    - DAGs are a data structure
    - Massively parallelize operations
    - Distribute to multiple execution environments
    - Transfer trained models
**Keras**
- What are the three main advantages of computational graphs?
  - Massively parallelize optations
  - Distribute to different execution environments
  - Transfer models
- Tansformation
  - Stateless: simple operations, such as averages, maximums, and data reshaping
  - Stateful: operations wih memory, which allows the layer to be trained
- Levels of Abstraction
  - Sequential Model: easiest to use, least exposed complexity, lowest amount of architecture flexibility.
```python
import tensorflow as tf

model = tf.keras.Sequential([
	tf.keras.layers.Dense(10)
	tf.keras.layers.Dense(1)
])
```
  - Funtional API: General pupose. Create elaborate models by treating layers as the functions they are. Good mix of customiazation and usability.
```python
import tensorflow as tf

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```
  - Model Subclassing: Tweak any part of the process you want, while not having to worry about the parts you don't.
```python
import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.dense1 = layers.Dense(10)
    self.dense2 = layers.Dense(1)
  def call(self, inputs):
    x = self.densse1(inputs)
    return self.dense2(x)
model = MyModel()
```
- 





