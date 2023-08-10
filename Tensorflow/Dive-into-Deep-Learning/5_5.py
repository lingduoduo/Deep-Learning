# 5.5. 读写文件

import numpy as np
import tensorflow as tf

# 5.5.1. 加载和保存张量
x = tf.range(4)
np.save('x-file.npy', x)

x2 = np.load('x-file.npy', allow_pickle=True)
print(x2)

y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
print((x2, y2))

mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
print(mydict2)


# 5.5.2. 加载和保存模型参数
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
net.save_weights('mlp.params')

clone = MLP()
clone.load_weights('mlp.params')

Y_clone = clone(X)
Y_clone == Y

