import tensorflow as tf


class BroadcasterModel(tf.keras.Model):

	def __init__(self, conf):
		super().__init__()

		self.broadcaster_embedding = tf.keras.Sequential(
			[
				tf.keras.layers.experimental.preprocessing.StringLookup(
					vocabulary = conf['unique_broadcasters'], mask_token = None
				),
				tf.keras.layers.Embedding(len(conf['unique_broadcasters']) + 1, conf['broadcaster_embedding_dimension'])
			]
		)

	def call(self, broadcaster):
		return tf.concat(
			[
				self.broadcaster_embedding(broadcaster),
			], axis = 1
		)


class CandidateModel(tf.keras.Model):
	"""Model for encoding movies."""

	def __init__(self, conf):
		"""Model for encoding movies.

		Args:
		  layer_sizes:
			A list of integers where the i-th entry represents the number of units
			the i-th layer contains.
		"""
		super().__init__()

		self.embedding_model = BroadcasterModel(conf)

		self.dense_layers = tf.keras.Sequential(
			[
				tf.keras.layers.Dense(32, activation = 'relu', kernel_regularizer = tf.keras.regularizers.L2(0.0001)),
				tf.keras.layers.Dropout(0.5),
				tf.keras.layers.Dense(32)
			]
		)

	def call(self, inputs):
		feature_embedding = self.embedding_model(inputs)
		return self.dense_layers(feature_embedding)
