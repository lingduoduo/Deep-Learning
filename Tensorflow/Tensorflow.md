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

**Deep Neural Network (DNN)**
- A Tensorflow NN is a graph of layers containing artificial neurons.
  - Input: Accepts raw data as tensors
  - Hidden: Intermediate transforamtions
  - Output: Tensor prediction
- Deep: 2+ vs Shallow: 1 layers
- Dense layers provide the core learning functionality
  - Computation limits: large numbers of neurons with lots of inputs take a long time to train
  - Overfitting: want to generalize, not memorize
  - Overengineering: simpler models will work nearly as well
```python
# tf.keras.layers.Dense(
#     units,
#     activation=None,
#     use_bias=True,
#     kernel_initializer='glorot_uniform',
#     bias_initializer='zeros',
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs
# )

# Create a `Sequential` model and add a Dense layer as the first layer.
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(16,)))
model.add(tf.keras.layers.Dense(32, activation='relu'))
# Now the model will take as input arrays of shape (None, 16)
# and output arrays of shape (None, 32).
# Note that after the first layer, you don't need to specify
# the size of the input anymore:
model.add(tf.keras.layers.Dense(32))
model.output_shape
```

**Convolutions**
- Convolution kernels have properties similar to tensors
  - shape: total areas to consider at once
  - stride: how far to move the kernal to see the next piece of data
- Convolutions Produce a New Representation per Dimension per Kernel
```python
# tf.keras.layers.Conv1D(
#     filters,
#     kernel_size,
#     strides=1,
#     padding='valid',
#     data_format='channels_last',
#     dilation_rate=1,
#     groups=1,
#     activation=None,
#     use_bias=True,
#     kernel_initializer='glorot_uniform',
#     bias_initializer='zeros',
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs
# )
# The inputs are 128-length vectors with 10 timesteps, and the batch size
# is 4.
input_shape = (4, 10, 128)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv1D(
32, 3, activation='relu',input_shape=input_shape[1:])(x)
print(y.shape)

# tf.keras.layers.Conv2D(
#     filters,
#     kernel_size,
#     strides=(1, 1),
#     padding='valid',
#     data_format=None,
#     dilation_rate=(1, 1),
#     groups=1,
#     activation=None,
#     use_bias=True,
#     kernel_initializer='glorot_uniform',
#     bias_initializer='zeros',
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs
# )

# The inputs are 28x28 RGB images with `channels_last` and the batch
# size is 4.
input_shape = (4, 28, 28, 3)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D(
2, 3, activation='relu', input_shape=input_shape[1:])(x)
print(y.shape)
```
**Pooling**
Max pooling selects the highest magnitude. Average pooling selects the mean magnitude.
```python
# tf.keras.layers.MaxPool1D(
#     pool_size=2,
#     strides=None,
#     padding='valid',
#     data_format='channels_last',
#     **kwargs
# )
x = tf.constant([1., 2., 3., 4., 5.])
x = tf.reshape(x, [1, 5, 1])
max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=2,
   strides=1, padding='valid')
max_pool_1d(x)

# tf.keras.layers.AveragePooling1D(
#     pool_size=2,
#     strides=None,
#     padding='valid',
#     data_format='channels_last',
#     **kwargs
# )
x = tf.constant([1., 2., 3., 4., 5.])
x = tf.reshape(x, [1, 5, 1])
avg_pool_1d = tf.keras.layers.AveragePooling1D(pool_size=2,
   strides=1, padding='valid')
avg_pool_1d(x)
```

**Convolutional Neural Network(CNN)**
- Flatten layer
- Patterns can appear anywhere in the data
- Convoluations are a weak form of memory about space
```python
# tf.keras.layers.Flatten(
#     data_format=None, **kwargs
# )
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, 3, 3, input_shape=(3, 32, 32)))
model.add(Flatten())
model.output_shape
```

**Recurrent Neural Network**
- Simple RNN memory is fuzzy, so not much better than convolutions
- Memory resets between sequences, each sequence is independent
- LSTM
  - Hold steady, what working memory that stores values unchanged 
  - Learn what's important, is the data being held worthwhile 
  - Forget what's not, out with the old, in with the new 
  
  - Forget gate: choose to forget or remember current long-term information
  - Input gate: choose to add new data from current input to long-term memory
  - Output gate: choose to apply long-term memory items to short term memory to affect output

  - Long term memory doesn't degrade
  - Memory works across very long sequences
  - More computationally expensive
- GRU
  - Reset gate: choose how much memory applies to current input
  - Update gate: choose how much memory to keep and how much current input to add
  - Memory is less robust
  - Simpler architecture provides faster training times
  - Research show GRUs offer competitive performance on some tasks, but not all
- Bidirectional RNN

**Loss Function**
- Backpropagation
- Loss function to determine how much the model differs from the ideal
- Supervised 
  - Regression to predict continuous data
    - Mean Absolute Error, MAE, Average of absolute difference of prediction and label, proportional cost
    - Mean Squared Error, MSE, Average of squared difference of prediction and label, exponential cost
      - Root Mean Squared Error, RMSE, Square root of MSE
    - Mean Absolute Percentage Error, MAPE, Average of the absolute difference of the prediction and label divided by the label, scale-agnostic cost
  - Classification to predict discrete data
    - Cross entropy
      - Binary cross entropy
      - Integer encoded multiclass, Sparse categorical entropy
      - Categorical one-hot encoded multiclass, categorical entropy
- Unsupervised
- Reinforcement

**Optimizer**
- Once we know the loss, how can we change the model to be better