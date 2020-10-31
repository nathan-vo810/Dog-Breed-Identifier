from model.model import Model
from network.redis import RedisClient
import time
import numpy as np
from multiprocessing import Process, Value
import pickle

class Classifier(Process):
	def __init__(self):
		super().__init__()
		self._redis = RedisClient()
		self._is_ready = Value('b', 0)

	def is_ready(self):
		return self._is_ready == 1

	def run(self):
		self.model = self._prepare_model()
		self.class_names = self._get_class_name()

		self._is_ready.value = 1
		print("Started Classifier!")

		while True:
			try:
				data = self._redis.pop_queue_obj("queue:identifier")
				if data is None:
					time.sleep(0.5)
				else:
					print("Received data!")
					probabilities, top5_indexes = self.model.predict(data)
					if probabilities is not None:
						self._return_result(probabilities, top5_indexes)
					else:
						self._redis.publish("identifier_result", "NO DOG!")
			except Exception as e:
				print(e)

	def _return_result(self, probabilities, top5_indexes):
		message = '\n'.join('{}: {}'.format(self.class_names[index], probabilities[index]) for index in top5_indexes if probabilities[index] > 1e-2)
		self._redis.publish("identifier_result", message)

	def _prepare_model(self):
		print("Loading model...")
		model = Model()
		model.load("./model/wide_resnet_883562.pth")
		print("Model loaded!")

		return model

	def _get_class_name(self):
		with open("class_names.npy", "rb") as f:
			class_names = np.load(f)
		return class_names


def start_process():
	process = Classifier()
	process.start()
	while not process.is_ready():
		pass

	return process