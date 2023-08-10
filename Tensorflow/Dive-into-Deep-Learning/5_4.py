# 5.4. 自定义层

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

