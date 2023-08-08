import tensorflow as tf


# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = tf.reshape(X, (1,) + X.shape + (1,))
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return tf.reshape(Y, Y.shape[1:3])


# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
print(comp_conv2d(conv2d, X).shape)

conv2d = tf.keras.layers.Conv2D(1, kernel_size=(5, 3), padding='same')
print(comp_conv2d(conv2d, X).shape)

conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
print(comp_conv2d(conv2d, X).shape)

conv2d = tf.keras.layers.Conv2D(1, kernel_size=(3, 5), padding='valid', strides=(3, 4))
print(comp_conv2d(conv2d, X).shape)
