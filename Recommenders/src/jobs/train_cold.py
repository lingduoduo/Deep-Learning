import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

from src.models.broadcaster import BroadcasterModel
from src.models.twotowers import TwoTowers
from src.models.user import QueryModel
from src.utils.log_helper import initialize_log
from src.utils.time_helper import log_time_cost

INPUT_FILE = os.path.dirname(__file__) + "/../../notebooks/csv/2021-11-22.csv"
OUTPUT_MODEL = os.path.dirname(__file__) + "../../save_models/models/jobs"


@log_time_cost
def load_data_file_cold(file):
	print('loading file:' + file)
	training_df = pd.read_csv(
		file,
		skiprows = [0],
		names = ["viewer",
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
		dtype = {
			'viewer': np.unicode,
			'broadcaster': np.unicode,
			'viewer_age': np.single,
			'viewer_gender': np.unicode,
			'viewer_longitude': np.single,
			'viewer_latitude': np.single,
			'viewer_lang': np.unicode,
			'viewer_country': np.unicode,
			'broadcaster_age': np.single,
			'broadcaster_longitude': np.single,
			'broadcaster_latitude': np.single,
			'broadcaster_lang': np.unicode,
			'broadcaster_country': np.unicode,
			'viewer_network': np.unicode,
			'broadcaster_network': np.unicode,
			'viewer_lat_long_cluster': np.unicode,
			'rank': np.unicode,
		}
	)

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
		"viewer_lat_long": tf.constant(["40.36393,-74.89611"]),
		'rank': '1'
	}
	# training_df = training_df.sample(frac = 0.0001)
	training_df.fillna(value = values, inplace = True)
	training_df['viewer_lat_long'] = training_df[['viewer_latitude', 'viewer_longitude']].apply(
		lambda x: '{},{}'.format(x[0], x[1]), axis = 1
	)
	print(training_df.head(10))
	print(training_df.iloc[-10:])
	# stats.send_stats('data-size', len(training_df.index))
	# training_df = training_df.sample(frac=0.1)
	return training_df


@log_time_cost
def load_training_data_cold(file):
	ratings_df = load_data_file_cold(file)
	print('creating data set')
	training_ds = (
		tf.data.Dataset.from_tensor_slices(
			({
				"viewer": tf.cast(
					ratings_df['viewer'].values,
					tf.string
				),
				"viewer_gender": tf.cast(
					ratings_df['viewer_gender'].values,
					tf.string
				),
				"viewer_lang": tf.cast(
					ratings_df['viewer_lang'].values,
					tf.string
				),
				"viewer_country": tf.cast(
					ratings_df['viewer_country'].values,
					tf.string
				),
				"viewer_age": tf.cast(
					ratings_df['viewer_age'].values,
					tf.int32
				),
				"viewer_longitude": tf.cast(
					ratings_df['viewer_longitude'].values,
					tf.float16
				),
				"viewer_latitude": tf.cast(
					ratings_df['viewer_latitude'].values,
					tf.float16
				),
				"broadcaster": tf.cast(
					ratings_df['broadcaster'].values,
					tf.string
				),
				"viewer_network": tf.cast(
					ratings_df['viewer_network'].values,
					tf.string
				),
				"broadcaster_network": tf.cast(
					ratings_df['broadcaster_network'].values,
					tf.string
				),
				"viewer_lat_long": tf.cast(
					ratings_df['viewer_lat_long'].values,
					tf.string
				),
			})
		))

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
			"viewer_lat_long": x["viewer_lat_long"],
		}, num_parallel_calls = tf.data.AUTOTUNE,
		deterministic = False
	)

	print('done prepare_training_data')
	return training_ds


def get_list(training_data, key):
	return training_data.batch(1_000_000).map(
		lambda x: x[key], num_parallel_calls = tf.data.AUTOTUNE, deterministic = False
	)


def get_unique_list(data):
	return np.unique(np.concatenate(list(data)))


def get_broadcaster_data_set(train_ds):
	broadcasters = train_ds.cache().map(
		lambda x: x["broadcaster"], num_parallel_calls = tf.data.AUTOTUNE, deterministic = False
	)
	broadcasters_ds = tf.data.Dataset.from_tensor_slices(
		np.unique(list(broadcasters.as_numpy_iterator()))
	)
	return broadcasters_ds


@log_time_cost
def tfrs_model_updates():
	logging.info("Start Training Job!")

	# Step 1: Load data
	training_dataset = load_training_data_cold(file = INPUT_FILE)
	print("Finish loading data")

	# Step 2: Pre-processing data
	train = prepare_training_data_cold(training_dataset)
	print("Finish pre-processing data")

	broadcasters_data_set = get_broadcaster_data_set(training_dataset)

	# Step 3: Prepare features
	user_genders = get_list(train, "viewer_gender")
	unique_user_genders = get_unique_list(user_genders)

	user_langs = get_list(train, "viewer_lang")
	unique_user_langs = get_unique_list(user_langs)

	user_countries = get_list(train, "viewer_country")
	unique_user_countries = get_unique_list(user_countries)

	user_networks = get_list(train, "viewer_network")
	unique_user_networks = get_unique_list(user_networks)

	broadcaster_ids = get_list(train, "broadcaster")
	unique_broadcasters = get_unique_list(broadcaster_ids)
	broadcaster_embedding_dimension = 32

	cold_start_conf = {
		'unique_genders': unique_user_genders,
		'unique_langs': unique_user_langs,
		'unique_countries': unique_user_countries,
		'unique_networks': unique_user_networks,
		'unique_broadcasters': unique_broadcasters,
		'broadcaster_embedding_dimension': broadcaster_embedding_dimension
	}
	print(cold_start_conf)
	print("Finish preparing features")

	# Step 4: Training routine
	hyperparameters = {
		"learning_rate": 0.05,
		"batch_size": 16384,
		"epochs": 2,
		"top_k": 1999,
		"patience": 2
	}
	query_model = QueryModel(cold_start_conf)
	candidate_model = BroadcasterModel(cold_start_conf)
	metrics = tfrs.metrics.FactorizedTopK(candidates = broadcasters_data_set.batch(128).map(candidate_model))
	task = tfrs.tasks.Retrieval(
		metrics = metrics
	)
	model = TwoTowers(candidate_model = candidate_model, query_model = query_model, task = task)
	model.compile(optimizer = tf.keras.optimizers.Adagrad(learning_rate = hyperparameters["learning_rate"]))

	tf.random.set_seed(42)
	shuffled = train.shuffle(100_000, seed = 42, reshuffle_each_iteration = True).repeat()
	train_p80 = shuffled.take(80_000)
	test_p20 = shuffled.skip(80_000).take(20_000)
	cached_train = train_p80.shuffle(100_000).batch(hyperparameters["batch_size"])
	cached_test = test_p20.batch(hyperparameters["batch_size"]).cache()

	callback = tf.keras.callbacks.EarlyStopping(
		monitor = 'total_loss',
		patience = hyperparameters["patience"],
		verbose = 1,
		restore_best_weights = True
	)
	model.fit(
		cached_train,
		epochs = hyperparameters["epochs"],
		validation_data = cached_test,
		validation_freq = 1,
		callbacks = [callback],
	)

	train_accuracy = model.evaluate(
		cached_train, return_dict = True
	)["factorized_top_k/top_100_categorical_accuracy"]
	test_accuracy = model.evaluate(
		cached_test, return_dict = True
	)["factorized_top_k/top_100_categorical_accuracy"]

	logging.info(f"Top-100 accuracy (train): {train_accuracy:.4f}.")
	logging.info(f"Top-100 accuracy (test): {test_accuracy:.4f}.")
	print("Finish Training Job!")


if __name__ == "__main__":
	initialize_log()
	tfrs_model_updates()
