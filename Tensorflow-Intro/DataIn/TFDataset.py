import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def printds(ds, quantity=5):
    for example in ds.take(quantity):
        print(example)

data = list(range(100))
ds = tf.data.Dataset.from_tensor_slices(data)
shuffled = ds.shuffle(buffer_size=10)
printds(shuffled, 10)
printds(shuffled)
print("End print results of buffer size = 10")

shuffled = ds.shuffle(buffer_size=20)
printds(shuffled, 10)
printds(shuffled)
print("End print results of buffer size = 20")

shuffled = ds.shuffle(buffer_size=50)
printds(shuffled, 10)
printds(shuffled)
print("End print results of buffer size = 50")

batched = ds.batch(10)
printds(batched, 10)
print("End print results of batch size = 10")

batched = ds.batch(12, drop_remainder=True)
printds(batched, 10)
print("End print results of batch size = 10")

batched = ds.batch(10, drop_remainder=True)
batched_shuffle = batched.shuffle(10)
printds(batched_shuffle, 10)
print("End print results of batched_shuffle  = 10")


shuffle_batch = shuffled.batch(10)
printds(shuffle_batch, 10)
print("End print results of batched_shuffle  = 10")

shuffled_batch = ds.shuffle(100).batch(10)
printds(shuffle_batch, 10)
print("End print results of batched_shuffle  = 10")

shuffled_batch = ds.shuffle(100).batch(10)
printds(shuffle_batch, 10)
print("End print results of batched_shuffle  = 10")

shuffled_batch = ds.shuffle(100).batch(10)
printds(shuffle_batch, 10)
print("End print results of batched_shuffle  = 10")

cached = shuffled_batch.cache()
printds(cached)
print("End print results of batched_shuffle  = 10")
printds(cached)
print("End print results of batched_shuffle  = 10")

cached = shuffled_batch.cache()
printds(cached, 10)
print("End print results of batched_shuffle  = 10")
printds(cached, 10)
print("End print results of batched_shuffle  = 10")

prefetch = ds.prefetch(tf.data.AUTOTUNE)
printds(prefetch)