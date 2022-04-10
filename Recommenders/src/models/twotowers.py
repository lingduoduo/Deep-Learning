import tensorflow as tf
from typing import Dict, Text


class TwoTowers(tf.keras.Model):

	def __init__(self, candidate_model, query_model, task):
		super().__init__()
		self.query_model: tf.keras.Model = query_model
		self.candidate_model: tf.keras.Model = candidate_model
		self.task = task

	def train_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
		# Set up a gradient tape to record gradients.
		with tf.GradientTape() as tape:
			# Loss computation.

			query_embeddings = self.query_model(
				{
					"viewer_gender": features["viewer_gender"],
					"viewer_lang": features["viewer_lang"],
					"viewer_country": features["viewer_country"],
					"viewer_age": features["viewer_age"],
					"viewer_network": features["viewer_network"],
					"viewer_latitude": features["viewer_latitude"],
					"viewer_longitude": features["viewer_longitude"],
					"viewer_lat_long": features["viewer_lat_long"],
				}
			)
			positive_broadcaster_embeddings = self.candidate_model(
				features["broadcaster"]
			)
			loss = self.task(query_embeddings, positive_broadcaster_embeddings)

			# Handle regularization losses as well.
			regularization_loss = sum(self.losses)

			total_loss = loss + regularization_loss

		gradients = tape.gradient(total_loss, self.trainable_variables)
		self.optimizer.apply_gradients(
			zip(gradients, self.trainable_variables)
		)

		metrics = {metric.name: metric.result() for metric in self.metrics}
		metrics["loss"] = loss
		metrics["regularization_loss"] = regularization_loss
		metrics["total_loss"] = total_loss

		return metrics

	def test_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
		# Loss computation.

		query_embeddings = self.query_model(
			{
				"viewer_gender": features["viewer_gender"],
				"viewer_lang": features["viewer_lang"],
				"viewer_country": features["viewer_country"],
				"viewer_age": features["viewer_age"],
				"viewer_network": features["viewer_network"],
				"viewer_latitude": features["viewer_latitude"],
				"viewer_longitude": features["viewer_longitude"],
				"viewer_lat_long": features["viewer_lat_long"],
			}
		)

		positive_broadcaster_embeddings = self.candidate_model(features["broadcaster"])
		loss = self.task(query_embeddings, positive_broadcaster_embeddings)

		# Handle regularization losses as well.
		regularization_loss = sum(self.losses)

		total_loss = loss + regularization_loss

		metrics = {metric.name: metric.result() for metric in self.metrics}
		metrics["loss"] = loss
		metrics["regularization_loss"] = regularization_loss
		metrics["total_loss"] = total_loss
		return metrics