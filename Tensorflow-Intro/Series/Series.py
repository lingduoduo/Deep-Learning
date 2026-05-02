import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gzip
import os
import shutil
from history import add_history, plot_history, save_history
import json

os.environ["TF_CPP_LOG_LEVEL"] = "2"


def retrieve_data():
    cache_dir = ".."
    cache_subdir = "data"
    traffic_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
    traffic_compressed = tf.keras.utils.get_file("traffic.csv.gz", traffic_url, cache_dir=cache_dir, cache_subdir=cache_subdir)

    out_file = f"{cache_dir}/{cache_subdir}/traffic.csv"
    if not os.path.exists(out_file) and not os.path.isfile(out_file):
        with gzip.open(traffic_compressed, 'rb') as t_in:
            with open(out_file, "wb") as t_out:
                shutil.copyfileobj(t_in, t_out)
    return out_file


def plot_sequence(time, sequences, start=0, end=None):
    y_max = 1.0
    if len(np.shape(sequences)) == 1:
        sequences = [sequences]
    time = time[start:end]
    plt.figure(figsize=(28, 8))
    for sequence in sequences:
        y_max = max(np.max(sequence), y_max)
        sequence = sequence[start:end]
        plt.plot(time, sequence)
    plt.show()


def split_data(traffic, time, split_size):
    d_split = int(np.ceil(len(traffic) * split_size))

    big_traffic = traffic[:d_split]
    big_time = time[:d_split]
    small_traffic = traffic[d_split:]
    small_time = time[d_split:]

    print(f"Splitting into {len(big_traffic)} and {len(small_traffic)} extra examples." )
    return big_traffic, big_time, small_traffic, small_time


def print_ds(ds, take=5):
    for ex in ds.take(take):
        print(ex)


def wrangle_data(sequence, data_split, examples, batch_size):
    examples = examples + 1
    seq_expand = tf.expand_dims(sequence, -1)
    dataset = tf.data.Dataset.from_tensor_slices(seq_expand)
    dataset = dataset.window(examples, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda  b: b.batch(examples))
    dataset = dataset.map(lambda x: (x[:-1], x[-1]))

    if data_split == "train":
        dataset = dataset.shuffle(10000)
    else:
        dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def compile_model(new_model, loss="mse"):
    new_model.compile(optimizer="adam", loss=loss, metrics=["mae"])
    print(new_model.summary())
    return new_model


def save_model(model, name, history, test_data):
    test_loss, test_acc = model.evaluate(test_data)

    save_name = f"models/cifar10-{name}-{len(history.epoch):02d}-{test_acc * 100:0.4f}"
    model.save(f"{save_name}.h5")

    # save history information
    hist_out = {}
    hist_out["epoch"] = history.epoch
    hist_out["history"] = history.history
    hist_out["params"] = history.params
    with open(f"{save_name}.history", "w") as outfile:
        json.dump(hist_out, outfile)


def dnn_model():
    new_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer((None, 1)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    return compile_model(new_model)


def cnn_model():
    new_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((None, 1)),
        tf.keras.layers.Conv1D(30, kernel_size=6, padding='causal', activation="relu"),
        tf.keras.layers.Conv1D(30, kernel_size=6, padding='causal', activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    return compile_model(new_model)


def rnn_model():
    new_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((None, 1)),
        tf.keras.layers.Conv1D(30, kernel_size=6, padding='causal', activation="relu"),
        tf.keras.layers.LSTM(60),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    return compile_model(new_model)


def stack_rnn_model():
    new_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((None, 1)),
        tf.keras.layers.Conv1D(30, kernel_size=6, padding='causal', activation="relu"),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.LSTM(60),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    return compile_model(new_model)


def show_predictions(trained_model, predict_sequence, true_values, predict_time, begin=0, end=None):
    predictions = trained_model.predict(predict_sequence)
    predictions = predictions[:, -1].reshape(len(predictions))
    plot_sequence(predict_time, (true_values, predictions), begin, end)
    return predictions


# def bigger_dnn_mode
if __name__ == "__main__":
    traffic_file = retrieve_data()

    # Load data
    traffic_df = pd.read_csv(traffic_file)
    traffic_volume = np.array(traffic_df["traffic_volume"])
    time_steps = np.array(list(range(len(traffic_volume))))

    # Extra variables for visualization
    day = 24
    week = 168
    month = 672

    traffic_df.head()
    traffic_df.describe()

    plot_sequence(time_steps, traffic_volume, end=week)
    plot_sequence(time_steps, traffic_volume, end=month)

    # Normalize the dataset
    max_traffic = np.max(traffic_volume)
    min_traffic = np.min(traffic_volume)
    traffic_normalized = (traffic_volume - min_traffic) / (max_traffic - min_traffic)

    train_volume_tmp, train_time_tmp, test_volume, test_time = split_data(traffic_normalized, time_steps, split_size=0.8)
    train_volume, train_time, valid_volume, valid_time = split_data(train_volume_tmp, train_time_tmp, 0.9)

    examples = 24
    batch_size = 30

    train_data = wrangle_data(train_volume, "train", examples, batch_size)
    valid_data = wrangle_data(valid_volume, "valid", examples, batch_size)
    test_data = wrangle_data(test_volume, "test", examples, batch_size)

    # model_name = "dnn"
    # earlystop = tf.keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=f"ckpts/traffic/{model_name}/" + "{epoch:02d}-{val_loss:.4f}")
    #
    # model = dnn_model()
    # history = model.fit(train_data, validation_data=valid_data, callbacks=[earlystop, checkpoint], epochs=10)
    # plot_history(history)
    # save_model = save_model(model, model_name, history, test_data)

    # model_name = "cnn"
    # earlystop = tf.keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=f"ckpts/traffic/{model_name}/" + "{epoch:02d}-{val_loss:.4f}")
    #
    # model = cnn_model()
    # model.summary()
    # history = model.fit(train_data, validation_data=valid_data, callbacks=[earlystop, checkpoint], epochs=10)
    # plot_history(history)
    # save_model = save_model(model, model_name, history, test_data)


    # model_name = "rnn"
    # earlystop = tf.keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=f"ckpts/traffic/{model_name}/" + "{epoch:02d}-{val_loss:.4f}")
    #
    # model = rnn_model()
    # model.summary()
    # history = model.fit(train_data, validation_data=valid_data, callbacks=[earlystop, checkpoint], epochs=10)
    # plot_history(history)
    # save_model = save_model(model, model_name, history, test_data)

    # model_name = "stack_rnn"
    # earlystop = tf.keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=f"ckpts/traffic/{model_name}/" + "{epoch:02d}-{val_loss:.4f}")
    #
    # model = stack_rnn_model()
    # model.summary()
    # history = model.fit(train_data, validation_data=valid_data, callbacks=[earlystop, checkpoint], epochs=10)
    # plot_history(history)
    # save_model = save_model(model, model_name, history, test_data)

    # dnn_model = tf.keras.models.load_model("models/cifar10-dnn-07-23.5459.h5")
    # dnn_model.evaluate(test_data)
    # show_predictions(dnn_model, test_data, test_volume[examples:], test_time[examples:])

    # cnn_model = tf.keras.models.load_model("models/cifar10-cnn-10-19.6308.h5")
    # cnn_model.evaluate(test_data)
    # show_predictions(cnn_model, test_data, test_volume[examples:], test_time[examples:])

    rnn_model = tf.keras.models.load_model("models/cifar10-stack_rnn-10-3.7421.h5")
    rnn_model.evaluate(test_data)
    show_predictions(rnn_model, test_data, test_volume[examples:], test_time[examples:])