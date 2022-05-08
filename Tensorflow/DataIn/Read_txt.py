import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if __name__ == "__main__":
    cache_dir = ".."
    cache_subdir = "data"

    # frank_url = "https:/.../tiny_frankenstein.tgz"
    # tf.keras.utils.get_file("tiny_frankenstein.tgz", frank_url, extract=True, cache_dir=cache_dir, cache_subdir=cache_subdir)

    frank_file = f"{cache_dir}/{cache_subdir}/tiny_frankenstein.tgz"
    frank_dataset = tf.data.TextLineDataset(frank_file)

    for example in frank_dataset.take(5):
        print(example)

    

