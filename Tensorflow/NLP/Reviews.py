import tensorflow as tf
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# from history import add_history, plot_history, save_history

os.environ["TF_CPP_LOG_LEVEL"] = "2"


def plot_sequence(time=[], sequence="", start=0, end=None):
    y_min, ymax = (0.0, 1.0)
    if len(np.shape(sequence)) == 1:
        sequence = [sequence]
    if len(time) == 0:
        time = range(len(sequence[0]))
    time = time[start:end]
    plt.figure(figzise=(28, 8))
    for seq in sequence:
        seq = seq[start:end]
        y_max = max(np.max(seq), y_max)
        y_min = min(np.min(seq), y_min)
        plt.plot(time, seq)
    plt.ylim(y_min, y_max)
    plt.xlim(np.min(time), np.max(time))


def print_dict(json_dict, items=5):
    print({x: json_dict[x] for (i, x) in enumerate(json_dict) if i < items})


def retrieve_data():
    cache_dir = ".."
    cache_subdir = "data"
    imdb_dir = f"{cache_dir}/{cache_subdir}/aclImdb"
    imdb_train_dataset = tf.keras.preprocessing.text_dataset_from_directory(
        f"{imdb_dir}/train",
        label_mode="binary",
        batch_size=1,
        seed=42
    )

    imdb_test_dataset = tf.keras.preprocessing.text_dataset_from_directory(
        f"{imdb_dir}/test",
        label_mode="binary",
        batch_size=1,
        shuffle=False
    )
    return imdb_train_dataset, imdb_test_dataset


def split_features_labels(dataset):
    dataset_raw = list(dataset.as_numpy_iterator())
    features, labels = zip(*dataset_raw)
    features = [x[0].decode("utf-8").lower() for x in features]
    labels = [float(x) for x in labels]
    return features, labels


def wrangle_data(tokenizer, features, labels, sequence_length):
    tokens = tokenizer.texts_to_sequences(features)
    features_padding = tf.keras.preprocessing.sequence.pad_sequences(tokens,
                                                                     maxlen=sequence_length,
                                                                     padding="post",
                                                                     truncating="post")
    labels = np.array(labels)
    return features_padding, labels, tokens


def compile_model(new_model):
    new_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print(new_model.summary)
    return new_model


def dnn_model(word_dim, embedding_dim, seq_length):
    new_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(word_dim, embedding_dim, input_length=seq_length),
        tf.keras.layers.GlobalAvgPool1D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    return compile_model(new_model)


def save_model(model, name, history, test_data, test_labels):
    test_loss, test_acc = model.evaluate(test_data, test_labels)

    save_name = f"models/reviews-{name}-{len(history.epoch):02d}-{test_acc:0.4f}"
    model.save(f"{save_name}.h5")

    # save_history(history, save_name)


if __name__ == "__main__":
    train_dataset, test_dataset = retrieve_data()
    train_features, train_labels = split_features_labels(train_dataset)
    test_features, test_labels = split_features_labels(test_dataset)

    word_dimenstion = 3000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=word_dimenstion, oov_token="---")
    tokenizer.fit_on_texts(train_features)

    sequence_length = 100
    train_data, train_labels, train_tokens = wrangle_data(tokenizer,
                                                          train_features,
                                                          train_labels,
                                                          sequence_length)
    test_data, test_labels, test_tokens = wrangle_data(tokenizer,
                                                       test_features,
                                                       test_labels,
                                                       sequence_length)
    model_name = "dnn"
    earlystop = tf.keras.callbacks.EarlyStopping('val_loss', patience=3, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'ckpts/reviews/{model_name}/' + "{epoch:02d}-{val_accuracy:.4f}")

    embedding_dimension = 32

    dnn = dnn_model(word_dimenstion, embedding_dimension, sequence_length)
    dnn.summary()
    history_dnn = dnn.fit(train_data, train_labels, callbacks=[earlystop, checkpoint], epochs=25, batch_size=64, validation_split=0.1)
    dnn.evaluate(test_data, test_labels)
    save_model(dnn, model_name, history_dnn, test_data, test_labels)
