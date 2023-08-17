# 5.4. 自定义层

# 在自定义层的时候继承了keras.Model基类，这个类中就包含build和call方法。这边相当于对build方法进行了重定义。
# build方法用于在首次传入input时进行权重的初始化，keras会自动调用 build 方法来创建层的变量。
# call方法是每一次进行前向传播计算时，Keras 会自动调用模型中每个层的 call 方法。
# 简单来说，就是在首次传入Inputs进行初始化的时候build方法被keras自动调用了，然后再执行call计算。tensorflow官方文档讲优点是： 单独实现 build() 很好地将只创建一次权重与在每次调用时使用权重分开。

import tensorflow as tf


class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)

layer = CenteredLayer()
print(layer(tf.constant([1, 2, 3, 4, 5])))

net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
Y = net(tf.random.uniform((4, 8)))
print(tf.reduce_mean(Y))

# 5.4.2. 带参数的层
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())

        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)

dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
print(dense.get_weights())

print(dense(tf.random.uniform((2, 5))))

net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
print(net(tf.random.uniform((2, 64))))

