import tensorflow as tf
import tensorflow_datasets as tfds
import os
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def retrieve_data():
    test_ds, cifar10_info = tfds.load("cifar10", split="test", with_info=True, as_supervised=True, shuffle_files=True)
    print(cifar10_info)
    print(f'classes:{cifar10_info.features["label"].names}')
    print(test_ds.element_spec)
    return test_ds, cifar10_info


def get_training_data(validation_split=10):
    validation_ds = tfds.load("cifar10", split=f"train[:{validation_split}%]", as_supervised=True)
    training_ds = tfds.load("cifar10", split=f"train[:{validation_split}%]", as_supervised=True)
    return training_ds, validation_ds


def wrangle_data(data, split, batch_size=32):
    data = data.map(lambda f, l: (tf.cast(f, tf.float64) / 255, l))

    if split == "train":
        data = data.shuffle(buffer_size=5000)
    elif split == "valid":
        data = data.cache()
    elif split == "test":
        data = data.cache()

    data = data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return data


if __name__ == "__main__":
    test_ds, image_info = retrieve_data()
    train_ds, valid_ds = get_training_data(validation_split=10)

    batch_size = 64
    train_data = wrangle_data(train_ds, "train", batch_size=batch_size)
    valid_data = wrangle_data(valid_ds, "valid", batch_size=batch_size)
    test_data = wrangle_data(test_ds, "test", batch_size=batch_size)

