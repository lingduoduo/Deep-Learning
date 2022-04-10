import os
import logging
import wandb

import pandas as pd
import numpy as np

from src.utils.time_helper import log_time_cost
from src.utils.log_helper import initialize_log
from src.models.lightgbm import build_model
from sklearn.model_selection import ParameterGrid

INPUT_FILE = os.path.dirname(__file__) + "/../../notebooks/csv/2021-11-05.csv"

FEATURES = [
	'broadcaster_id',
	'viewer_network',
	'viewer_age_bucket',
	'viewer_gender',
	'viewer_lang',
	'viewer_country',
	'viewer_lat_long_cluster'
]

def age_bucket(age):
	# min(age) = 18
	if age < 25:
		return 0
	elif age < 30:
		return 1
	elif age < 35:
		return 2
	elif age < 40:
		return 3
	elif age < 45:
		return 4
	elif age < 50:
		return 5
	elif age < 55:
		return 6
	elif age < 60:
		return 7
	elif age < 65:
		return 8
	else:
		return 9

@log_time_cost
def lightgbm_tuning_cutoff():
	print(INPUT_FILE)
	logging.info("Start Training Job!")

	# Step 1: Load data
	df = pd.read_csv(INPUT_FILE)
	# df = df.sample(frac=0.01)

	# Step 2: Pre-processing data
	df.dropna(inplace = True)
	logging.info(df.info())

	# Step 3: Prepare features
	df['viewer_age_bucket'] = df['viewer_age'].apply(lambda x: age_bucket(x))
	cutoffs = [df['duration'].quantile(c) for c in [.9, 0.95, .99]]

	res = pd.DataFrame()
	for cutoff in cutoffs:
		df_cp = df.copy()
		df_cp['duration'] = df_cp['duration'].apply(lambda x: cutoff if x > cutoff else x)
		# Step 4: Training routine
		df_agg = (df_cp.groupby(FEATURES)
		          .agg({'duration': np.mean, 'viewer_id': np.size})
		          .reset_index()
		          .rename(columns = {'viewer_id': 'weight'}))
		df_agg['log_duration'] = df_agg['duration'].apply(lambda x: np.log(x))
		print(f"Data {cutoff} - df_agg Done")

		pred_model, pred_metadata, train_df, test_df = build_model(
			X = df_agg[FEATURES],
			y = df_agg['log_duration'],
			w = df_agg['weight'],
			categories = FEATURES,
			unencoded_categories = FEATURES,
			params = {'n_estimators': 300, 'num_leaves': 300, "learning_rate": 0.1, 'max_bin': 100},
			regression = True
		)
		r = {'cutoff': cutoff}
		train_res = {**r, **pred_metadata['train_accuracy']}
		res = pd.concat([res, pd.DataFrame(train_res)], ignore_index=True)
		r = {'cutoff': cutoff}
		test_res = {**r, **pred_metadata['test_accuracy']}
		res = pd.concat([res, pd.DataFrame(test_res)], ignore_index = True)
		print(f"{cutoff} Done")
	res.to_csv("res.csv", index=False)

@log_time_cost
def lightgbm_tuning_hyperparameters():
	print(INPUT_FILE)
	logging.info("Start Training Job!")

	# Step 1: Load data
	df = pd.read_csv(INPUT_FILE)
	# df = df.sample(frac=0.001)

	# Step 2: Pre-processing data
	df.dropna(inplace = True)
	logging.info(df.info())

	# Step 3: Prepare features
	df['viewer_age_bucket'] = df['viewer_age'].apply(lambda x: age_bucket(x))
	cutoff = df['duration'].quantile(0.99)

	df_cp = df.copy()
	df_cp['duration'] = df_cp['duration'].apply(lambda x: cutoff if x > cutoff else x)
	# Step 4: Training routine
	df_agg = (df_cp.groupby(FEATURES)
	          .agg({'duration': np.mean, 'viewer_id': np.size})
	          .reset_index()
	          .rename(columns = {'viewer_id': 'weight'}))
	df_agg['log_duration'] = df_agg['duration'].apply(lambda x: np.log(x))

	tuning_conf = {
		"learning_rate": [0.01, 0.05, 0.1],
		"n_estimators": [250, 400, 450],
		"num_leaves": [100, 300, 500],
		"max_bin": [100, 250, 400]
	}

	df_cp = df.copy()
	df_cp['duration'] = df_cp['duration'].apply(lambda x: cutoff if x > cutoff else x)
	# Step 4: Training routine
	df_agg = (df_cp.groupby(FEATURES)
	          .agg({'duration': np.mean, 'viewer_id': np.size})
	          .reset_index()
	          .rename(columns = {'viewer_id': 'weight'}))
	df_agg['log_duration'] = df_agg['duration'].apply(lambda x: np.log(x))
	print(f"Data {cutoff} - df_agg Done")

	res = pd.DataFrame()
	for conf in ParameterGrid(tuning_conf):
		pred_model, pred_metadata, train_df, test_df = build_model(
			X = df_agg[FEATURES],
			y = df_agg['log_duration'],
			w = df_agg['weight'],
			categories = FEATURES,
			unencoded_categories = FEATURES,
			params = conf,
			regression = True
		)
		r = {'conf': conf}
		train_res = {**r, **pred_metadata['train_accuracy']}
		res = pd.concat([res, pd.DataFrame(train_res)], ignore_index=True)
		r = {'conf': conf}
		test_res = {**r, **pred_metadata['test_accuracy']}
		res = pd.concat([res, pd.DataFrame(test_res)], ignore_index = True)
		print(f"{conf} Done")
	res.to_csv("res.csv", index=False)

if __name__ == "__main__":
	initialize_log()
	# lightgbm_tuning_cutoff()
	lightgbm_tuning_hyperparameters()
