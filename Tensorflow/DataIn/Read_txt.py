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

    # imdb_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    # tf.keras.utils.get_file("aclImdb_v1.tar.gz", imdb_url, extract=True, cache_dir=cache_dir, cache_subdir=cache_subdir)

    imdb_dir = f"{cache_dir}/{cache_subdir}/aclImdb"
    imdb_train_dataset = tf.keras.preprocessing.text_dataset_from_directory(
        f"{imdb_dir}/train",
        label_mode="binary",
        batch_size=32,
        seed=42
    )

    for example, label in imdb_train_dataset.take(1):
        print(example.numpy()[0])
        print(label[0])
