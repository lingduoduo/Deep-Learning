import os
import logging

import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

from src.utils.time_helper import log_time_cost
from src.utils.log_helper import initialize_log

from sklearn.model_selection import ParameterGrid

from src.models.user import QueryModel
from src.models.broadcaster import BroadcasterModel
from src.models.twotowers import TwoTowers
from src.jobs.train_cold import load_training_data_cold, prepare_training_data_cold, get_broadcaster_data_set, get_list,\
	get_unique_list

INPUT_FILE = os.path.dirname(__file__) + "/../../notebooks/csv/2021-11-22.csv"

@log_time_cost
def tfrs_model_tuning_parameters():
	logging.info("Start Training Job!")

	# Step 1: Load data
	df = load_training_data_cold(file = INPUT_FILE)

	# Step 2: Pre-processing data
	train = prepare_training_data_cold(df)
	broadcasters_data_set = get_broadcaster_data_set(df)
	print("Finish pre-processing data")

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
	query_model = QueryModel(cold_start_conf)
	candidate_model = BroadcasterModel(cold_start_conf)
	metrics = tfrs.metrics.FactorizedTopK(candidates = broadcasters_data_set.batch(128).map(candidate_model))
	task = tfrs.tasks.Retrieval(
		metrics = metrics
	)
	model = TwoTowers(candidate_model = candidate_model, query_model = query_model, task = task)

	tuning_conf = {
		"learning_rate": [0.05, 0.1],
		"batch_size": [16384],
		"epochs": [20],
		"top_k": [1999]
	}
	tf.random.set_seed(42)
	shuffled = train.shuffle(100_000, seed = 42, reshuffle_each_iteration = True).repeat()
	train_p80 = shuffled.take(80_000)
	test_p20 = shuffled.skip(80_000).take(20_000)

	optimal_accuracy = 0
	optimal_conf = {}
	file = open("tmp.txt", "w")
	for conf in ParameterGrid(tuning_conf):
		cached_train = train_p80.shuffle(100_000).batch(conf["batch_size"])
		cached_test = test_p20.batch(conf["batch_size"]).cache()
		model.compile(optimizer = tf.keras.optimizers.Adagrad(learning_rate = conf["learning_rate"]))

		callback = tf.keras.callbacks.EarlyStopping(
			monitor = 'total_loss',
			patience = 2,
			verbose = 1,
			restore_best_weights = True
		)
		model.fit(
			cached_train,
			epochs = conf["epochs"],
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

		if optimal_accuracy < train_accuracy:
			optimal_accuracy = train_accuracy
			optimal_conf = conf

		print([conf, train_accuracy, test_accuracy])
		file.writelines(str(conf) + str(train_accuracy) + str(test_accuracy) + "\n")

	logging.info(
		{
			"optimal conf": optimal_conf,
			"Top-100 optimal accuracy (train)": f"{optimal_accuracy:.2f}",
		}
	)
	file.close()
	logging.info("Finish Hyperparameter Tuning Job!")


if __name__ == "__main__":
	initialize_log()
	tfrs_model_tuning_parameters()
