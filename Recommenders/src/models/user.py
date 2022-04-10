import numpy as np
import tensorflow as tf


class UserModel(tf.keras.Model):

	def __init__(self, conf):
		super().__init__()

		self.gender_embedding = tf.keras.Sequential(
			[
				tf.keras.layers.experimental.preprocessing.StringLookup(
					vocabulary = conf['unique_genders'], mask_token = None
				),
				tf.keras.layers.Embedding(len(conf['unique_genders']) + 1, 4),
			]
		)

		self.lang_embedding = tf.keras.Sequential(
			[
				tf.keras.layers.experimental.preprocessing.StringLookup(
					vocabulary = conf['unique_langs'], mask_token = None
				),
				tf.keras.layers.Embedding(len(conf['unique_langs']) + 1, 10),
			]
		)

		self.country_embedding = tf.keras.Sequential(
			[
				tf.keras.layers.experimental.preprocessing.StringLookup(
					vocabulary = conf['unique_countries'], mask_token = None
				),
				tf.keras.layers.Embedding(len(conf['unique_countries']) + 1, 10),
			]
		)

		self.network_embedding = tf.keras.Sequential(
			[
				tf.keras.layers.experimental.preprocessing.StringLookup(
					vocabulary = conf['unique_networks'], mask_token = None
				),
				tf.keras.layers.Embedding(len(conf['unique_networks']) + 1, 4),
			]
		)

		age_boundaries = np.array([18, 25, 30, 35, 40, 45, 50, 55, 60, 65, float("inf")])
		self.viewer_age_embedding = tf.keras.Sequential(
			[
				tf.keras.layers.experimental.preprocessing.Discretization(age_boundaries.tolist()),
				tf.keras.layers.Embedding(len(age_boundaries), 2)
			]
		)

		self.centroids = tf.constant(
			[
				[36.68147669256268, -82.8910274009993],
				[23.22243322909555, 78.23027450833709],
				[50.04997682638993, 0.22379313938744885],
				[37.9309447099281, -117.00741350764692],
				[-32.795864819917725, 148.7159172660312],
				[-18.570548393114084, -54.280255665692565],
				[13.921140442819565, 116.38740315555172],
				[29.78951080730802, 40.279515865947936]]
		)
		self.viewer_lat_long_embedding = tf.keras.Sequential(
			[
				tf.keras.layers.experimental.preprocessing.TextVectorization(
					standardize = None, split = self.classify,
					vocabulary = [str(i) for i in range(len(self.centroids))],
					max_tokens = len(self.centroids) + 2
				),
				tf.keras.layers.Embedding(len(self.centroids) + 2, 2),
			]
		)

	@tf.function()
	def call(self, inputs):
		return tf.concat(
			[
				self.gender_embedding(inputs["viewer_gender"]),
				self.lang_embedding(inputs["viewer_lang"]),
				self.country_embedding(inputs["viewer_country"]),
				self.network_embedding(inputs["viewer_network"]),
				self.viewer_age_embedding(inputs["viewer_age"]),
				self.viewer_lat_long_embedding(inputs["viewer_lat_long"]),
			], axis = 1
		)

	@tf.keras.utils.register_keras_serializable()
	def classify(self, pair):
		"""
		given a datapoint, compute the cluster closest to the datapoint. Return the cluster ID of that cluster.
		:param pair:
		:return: cluster ID
		"""
		str_data = tf.strings.split(pair, sep = ",").values
		str_data = tf.map_fn(lambda x: tf.strings.regex_replace(x, "b'", ""), str_data)
		datapoints = tf.map_fn(lambda x: tf.strings.to_number(x), str_data, dtype = (tf.float32))
		datapoints = tf.reshape(datapoints, [-1, 2])

		expanded_centroids = tf.expand_dims(self.centroids, 1)
		expanded_vectors = tf.expand_dims(datapoints, 0)
		distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
		clusters = tf.math.argmin(distances)
		return tf.strings.as_string(clusters)


class QueryModel(tf.keras.Model):
	"""Model for encoding user queries."""

	def __init__(self, conf):
		"""Model for encoding user queries.

		Args:
		  layer_sizes:
			A list of integers where the i-th entry represents the number of units
			the i-th layer contains.
		"""
		super().__init__()

		# We first use the user model for generating embeddings.
		self.embedding_model = UserModel(conf)
		self.dense_layers = tf.keras.Sequential(
			[
				tf.keras.layers.Dense(32, activation = 'relu', kernel_regularizer = tf.keras.regularizers.L2(0.0001)),
				tf.keras.layers.Dense(32)
			]
		)

	def call(self, inputs):
		feature_embedding = self.embedding_model(inputs)
		return self.dense_layers(feature_embedding)
