import os
import logging
import time

from src.utils.time_helper import log_time_cost
from src.utils.log_helper import initialize_log

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_recommenders as tfrs

from src.models.user import UserModel
from src.models.broadcaster import BroadcasterModel
from src.models.twotowers import TwoTowers

INPUT_FILE = os.path.dirname(__file__) + "/../../save_models/csv/2021-10-05.csv"
OUTPUT_MODEL = os.path.dirname(__file__) + "../../save_models/models/debugs"


def get_broadcaster_data_set(train_ds):
    broadcasters = train_ds.cache().map(
        lambda x: x["broadcaster"], num_parallel_calls = tf.data.AUTOTUNE, deterministic = False
    )
    broadcasters_ds = tf.data.Dataset.from_tensor_slices(
        np.unique(list(broadcasters.as_numpy_iterator()))
    )
    return broadcasters_ds


def get_list(training_data, key):
    return training_data.batch(1_000_000).map(
        lambda x: x[key], num_parallel_calls = tf.data.AUTOTUNE, deterministic = False
    )


def get_unique_list(data):
    return np.unique(np.concatenate(list(data)))


def load_data_file_cold(file, stats):
    print('loading file:' + file)
    training_df = pd.read_csv(
        file,
        skiprows=[0],
        names=["viewer",
               "broadcaster",
               "viewer_age",
               "viewer_gender",
               "viewer_longitude",
               "viewer_latitude",
               "viewer_lang",
               "viewer_country",
               "broadcaster_age",
               "broadcaster_gender",
               "broadcaster_longitude",
               "broadcaster_latitude",
               "broadcaster_lang",
               "broadcaster_country",
               "duration",
               "viewer_network",
               "broadcaster_network",
               "viewer_lat_long_cluster",
               "rank"],
        dtype={
            'viewer': np.unicode,
            'broadcaster': np.unicode,
            'viewer_age': np.single,
            'viewer_gender': np.unicode,
            'viewer_longitude': np.single,
            'viewer_latitude': np.single,
            'viewer_lang': np.unicode,
            'viewer_country': np.unicode,
            'broadcaster_age': np.int,
            'broadcaster_longitude': np.single,
            'broadcaster_latitude': np.single,
            'broadcaster_lang': np.unicode,
            'broadcaster_country': np.unicode,
            'viewer_network': np.unicode,
            'broadcaster_network': np.unicode,
            'viewer_lat_long_cluster': np.unicode,
            'rank': np.int
        })

    values = {
        'viewer': 'unknown',
        'broadcaster': 'unknown',
        'viewer_age': 30,
        'viewer_gender': 'unknown',
        'viewer_longitude': 0,
        'viewer_latitude': 0,
        'viewer_lang': 'unknown',
        'viewer_country': 'unknown',
        'broadcaster_age': 30,
        'broadcaster_longitude': 0,
        'broadcaster_latitude': 0,
        'broadcaster_lang': 'unknown',
        'broadcaster_country': 'unknown',
        'duration': 0,
        'viewer_network': 'unknown',
        'broadcaster_network': 'unknown',
        'viewer_lat_long_cluster': '0',
        'rank': 1
    }
    training_df.fillna(value=values, inplace=True)
    print(training_df.head(10))
    print(training_df.iloc[-10:])
    # stats.send_stats('data-size', len(training_df.index))
    samples = training_df.sample(frac=.1)
    return samples


def load_training_data_cold(file, stats):
    ratings_df = load_data_file_cold(file, stats)
    print('creating data set')
    training_ds = (
        tf.data.Dataset.from_tensor_slices(
            ({
                "viewer": tf.cast(
                    ratings_df['viewer'].values,
                    tf.string),
                "viewer_gender": tf.cast(
                    ratings_df['viewer_gender'].values,
                    tf.string),
                "viewer_lang": tf.cast(
                    ratings_df['viewer_lang'].values,
                    tf.string),
                "viewer_country": tf.cast(
                    ratings_df['viewer_country'].values,
                    tf.string),
                "viewer_age": tf.cast(
                    ratings_df['viewer_age'].values,
                    tf.int32),
                "viewer_longitude": tf.cast(
                    ratings_df['viewer_longitude'].values,
                    tf.float16),
                "viewer_latitude": tf.cast(
                    ratings_df['viewer_latitude'].values,
                    tf.float16),
                "broadcaster": tf.cast(
                    ratings_df['broadcaster'].values,
                    tf.string),
                "viewer_network": tf.cast(
                    ratings_df['viewer_network'].values,
                    tf.string),
                "broadcaster_network": tf.cast(
                    ratings_df['broadcaster_network'].values,
                    tf.string),
                "viewer_lat_long_cluster": tf.cast(
                    ratings_df['viewer_lat_long_cluster'].values,
                    tf.string),
            })))

    return training_ds


def prepare_training_data_cold(train_ds):
    print('prepare_training_data')
    training_ds = train_ds.cache().map(
        lambda x: {
            "broadcaster": x["broadcaster"],
            "viewer": x["viewer"],
            "viewer_gender": x["viewer_gender"],
            "viewer_lang": x["viewer_lang"],
            "viewer_country": x["viewer_country"],
            "viewer_age": x["viewer_age"],
            "viewer_longitude": x["viewer_longitude"],
            "viewer_latitude": x["viewer_latitude"],
            "viewer_network": x["viewer_network"],
            "broadcaster_network": x["broadcaster_network"],
            "viewer_lat_long_cluster": x["viewer_lat_long_cluster"],
        }, num_parallel_calls = tf.data.AUTOTUNE,
        deterministic = False
    )

    print('done prepare_training_data')
    return training_ds


def current_milli_time():
    return round(time.time() * 1000)


def training_process_cold(training_data_file, model_save_name, stats):
    print("training_process")
    start_time = current_milli_time()

    broadcaster_embedding_dimension = 32
    viewer_embedding_dimension = 32

    batch_size = 1638
    learning_rate = 0.05
    epochs = 1
    top_k = 199

    training_dataset = load_training_data_cold(training_data_file, stats)
    train = prepare_training_data_cold(training_dataset)
    broadcasters_data_set = get_broadcaster_data_set(training_dataset)

    print("get lists")
    print(current_milli_time() - start_time)
    user_genders = get_list(train, "viewer_gender")
    user_langs = get_list(train, "viewer_lang")
    user_countries = get_list(train, "viewer_country")
    user_networks = get_list(train, "viewer_network")
    user_clusters = get_list(train, "viewer_lat_long_cluster")

    viewer_age = get_list(train, "viewer_age")
    viewer_longitude = get_list(train, "viewer_longitude")
    viewer_latitude = get_list(train, "viewer_latitude")

    broadcaster_ids = get_list(train, "broadcaster")

    data_set_size = len(broadcaster_ids)

    print(data_set_size)

    if data_set_size == 0:
        time.sleep(600)
        return

    print("get_unique_list")
    print(current_milli_time() - start_time)
    unique_broadcasters = get_unique_list(broadcaster_ids)
    unique_user_genders = get_unique_list(user_genders)
    unique_user_langs = get_unique_list(user_langs)
    unique_user_countries = get_unique_list(user_countries)
    unique_user_networks = get_unique_list(user_networks)
    print(unique_broadcasters)
    unique_user_clusters = get_unique_list(user_clusters)
    print(unique_user_clusters)
    print(current_milli_time() - start_time)

    # stats.send_stats("unique_broadcasters", len(unique_broadcasters))

    print("unique broadcasters: " + str(len(unique_broadcasters)))

    print("create model")
    user_model = user_model = UserModel(
        unique_user_genders, unique_user_langs, unique_user_countries, viewer_age, unique_user_networks,
        unique_user_clusters
    )
    broadcaster_model = BroadcasterModel(
        unique_broadcasters,
        broadcaster_embedding_dimension
    )

    metrics = tfrs.metrics.FactorizedTopK(
        candidates = broadcasters_data_set.batch(128).map(broadcaster_model)
    )

    task = tfrs.tasks.Retrieval(
        metrics = metrics
    )

    model = TwoTowers(broadcaster_model, user_model, task)
    model.compile(
        optimizer = tf.keras.optimizers.Adagrad(
            learning_rate = learning_rate
        )
    )
    train_ds = train.batch(batch_size).cache()
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    print("train model")

    model.fit(
        train_ds,
        epochs = epochs
    )

    print("create index")
    index = tfrs.layers.factorized_top_k.BruteForce(
        query_model = user_model,
        k = top_k,
    )

    tf.random.set_seed(42)
    shuffled = train.shuffle(100_000, seed = 42, reshuffle_each_iteration = False)

    train_p80 = shuffled.take(80_000)
    test_p20 = shuffled.skip(80_000).take(20_000)

    cached_train = train_p80.shuffle(100_000).batch(2048)
    cached_test = test_p20.batch(2048).cache()

    new_model_history = model.fit(
        cached_train,
        validation_data = cached_test,
        validation_freq = 5,
        epochs = 10,
        verbose = 0
    )
    print(new_model_history.history)

    print("create index")
    index = tfrs.layers.factorized_top_k.BruteForce(
        query_model = user_model,
        k = top_k,
    )

    index.index(
        broadcasters_data_set.batch(10000).map(
            model.broadcaster_model
        ),
        broadcasters_data_set
    )

    # Get recommendations.
    _, broadcasters = index(
        {
            "viewer_gender": tf.constant(["female"]),
            "viewer_lang": tf.constant(["en"]),
            "viewer_country": tf.constant(["US"]),
            "viewer_age": tf.constant([32]),
            "viewer_longitude": tf.constant([-74.89611]),
            "viewer_latitude": tf.constant([40.36393]),
            "viewer_network": tf.constant(["skout"]),
            "viewer_lat_long_cluster": tf.constant(["7"]),
        }
    )
    print(f"Recommendations for user cal: {broadcasters}")

    index.save(model_save_name)


if __name__ == "__main__":
    initialize_log()
    training_process_cold(training_data_file = INPUT_FILE, model_save_name = OUTPUT_MODEL, stats = "")
