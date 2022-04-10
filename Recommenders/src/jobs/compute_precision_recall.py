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

from src.jobs.train_cold import load_data_file_cold

INPUT_FILE = os.path.dirname(__file__) + "/../../notebooks/csv/2021-10-05.csv"
OUTPUT_MODEL = os.path.dirname(__file__) + "../../save_models/models/debugs"

def pred_prob(new_df, index, k = 199):
	print(f"len(new_df): {len(new_df)}")
	predprob = []
	df = pd.DataFrame(
		new_df, columns = ['viewer_age',
		                   'viewer_country',
		                   'viewer_gender',
		                   'viewer_lang',
		                   'viewer_lat_long_cluster',
		                   'viewer_latitude',
		                   'viewer_longitude',
		                   'viewer_network',
		                   'broadcaster'
		                   ]
		)
	## score dim is decided when building model
	for idx, _ in df.iterrows():
		score, pred = index(
			{
				"viewer_gender": tf.constant([df.at[idx, "viewer_gender"]]),
				"viewer_lang": tf.constant([df.at[idx, "viewer_lang"]]),
				"viewer_country": tf.constant([df.at[idx, "viewer_country"]]),
				"viewer_age": tf.constant([int(df.at[idx, "viewer_age"])]),
				"viewer_longitude": tf.constant([df.at[idx, "viewer_longitude"]]),
				"viewer_latitude": tf.constant([df.at[idx, "viewer_latitude"]]),
				"viewer_network": tf.constant([df.at[idx, "viewer_network"]]),
				"viewer_lat_long_cluster": tf.constant([str(df.at[idx, "viewer_lat_long_cluster"])]),
			}
		)

		score = score.numpy()[0]
		k = min(k, len(score))
		### using normalization over topk scores to get probabilities and take the expectations as predictions
		normalized = [float(s) for s in score[:k]]
		predprob.append(np.average(np.arange(k) / k, weights = normalized))
	return np.mean(predprob)

def compute_precision_recall(model_path, data_path):
	index = tf.saved_model.load(model_path)
	# Pass a user id in, get top predicted movie titles back.
	score, broadcasters = index(
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
	print(broadcasters)
	print(score)

	## Precision and Recall
	samples = load_data_file_cold(data_path)
	samples = samples.sample(frac=.01)
	print(samples.info())

	res = pred_prob(samples, index, k = 199)
	print(res)

if __name__ == "__main__":
	initialize_log()
	compute_precision_recall(model_path = OUTPUT_MODEL, data_path = INPUT_FILE)
