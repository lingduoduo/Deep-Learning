import tensorflow as tf
import tensorflow_datasets as tfds
import os
import json
import numpy as np

from history import save_history, load_history, plot_history, add_history

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
        features = np.array([x[0] for x in data])
        labels = np.array([x[1] for x in data])
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip=True,
            zoom_range=0.2,
            rotation_range=20,
            fill_mode='nearest'
        )
        data = train_datagen.flow(features, labels, batch_size=batch_size)
    elif split in ("valid",  "test"):
        data = data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return data


def dnn_model():
    new_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((32, 32, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    return compile_model(new_model)


def compile_model(new_model):
    new_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(new_model.summary)
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


def cnn_model():
    new_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((32,32,3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    return compile_model(new_model)


def two_cnn_model():
    new_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((32,32,3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    return compile_model(new_model)


def three_cnn_model_augment():
    new_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((32,32,3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(258, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    return compile_model(new_model)


def three_cnn_model():
    new_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((32,32,3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    return compile_model(new_model)

if __name__ == "__main__":
    test_ds, image_info = retrieve_data()
    train_ds, valid_ds = get_training_data(validation_split=10)

    batch_size = 64
    train_data = wrangle_data(train_ds, "train", batch_size=batch_size)
    valid_data = wrangle_data(valid_ds, "valid", batch_size=batch_size)
    test_data = wrangle_data(test_ds, "test", batch_size=batch_size)

    # Prepare the model
    model_name= "dnn"
    model = dnn_model()

    # Create training callbacks
    earlystop = tf.keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=f"ckpts/cifar10-{model_name}-" + "{epoch:02d}-{val_accuracy:.4f}")

    # Train the model
    history = model.fit(train_data, validation_data=valid_data, epochs=10, callbacks=[earlystop, checkpoint])
    plot_history(history)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_data)
    print(f"Test accuray: {test_acc * 100: .2f}%")

    more_history = model.fit(train_data, validation_data=valid_data, callbacks=[earlystop, checkpoint], initial_epoch=10, epochs=20)
    add_history(history, more_history)

    # Save model information
    save_model(model, model_name, history, test_data)

    model_name = "cnn"
    model = cnn_model()

    # Create training callbacks
    earlystop = tf.keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=f"ckpts/cifar10-{model_name}-" + "{epoch:02d}-{val_accuracy:.4f}")

    # Train the model
    history = model.fit(train_data, validation_data=valid_data, epochs=10, callbacks=[earlystop, checkpoint])
    plot_history(history)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_data)
    print(f"Test accuray: {test_acc * 100: .2f}%")

    more_history = model.fit(train_data, validation_data=valid_data, callbacks=[earlystop, checkpoint], initial_epoch=10, epochs=20)
    add_history(history, more_history)

    better_model = tf.keras.models.load_model("ckpts/cifar-10-cnn-04-0.8176")
    better_model.evaluate(test_data)

    # Save model information
    save_model(better_model, "cnn-ck4", history, test_data)

    model_name = "2cnn"
    model = two_cnn_model()

    # Create training callbacks
    earlystop = tf.keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=f"ckpts/cifar10-{model_name}-" + "{epoch:02d}-{val_accuracy:.4f}")

    # Train the model
    history = model.fit(train_data, validation_data=valid_data, epochs=10, callbacks=[earlystop, checkpoint])
    plot_history(history)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_data)
    print(f"Test accuray: {test_acc * 100: .2f}%")

    more_history = model.fit(train_data, validation_data=valid_data, callbacks=[earlystop, checkpoint], initial_epoch=10, epochs=20)
    add_history(history, more_history)

    # Save model information
    save_model(model, "two-cnn", history, test_data)

    model_name = "three_cnn_model_augment"
    model = three_cnn_model()

    # Create training callbacks
    earlystop = tf.keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=f"ckpts/cifar10-{model_name}-" + "{epoch:02d}-{val_accuracy:.4f}")

    # Train the model
    history = model.fit(train_data, validation_data=valid_data, epochs=25, callbacks=[earlystop, checkpoint])
    plot_history(history)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_data)
    print(f"Test accuray: {test_acc * 100: .2f}%")

    more_history = model.fit(train_data, validation_data=valid_data, callbacks=[earlystop, checkpoint], initial_epoch=10, epochs=20)
    add_history(history, more_history)

    # Save model information
    save_model(model, "three-cnn", history, test_data)
