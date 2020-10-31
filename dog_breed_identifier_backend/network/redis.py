import pickle
import redis

class RedisClient:
	def __init__(self, host='redis', port=6379):
		self._redis = redis.StrictRedis(host, port)

	def set_string(self, key, value):
		self._redis.set(key, value)

	def get_string(self, key, default=None):
		raw_value = self._redis.get(key)
		if raw_value is None:
			return default
		return raw_value.decode('utf-8')

	def push_queue_obj(self, queue_name, obj):
		obj_bytes = pickle.dumps(obj)
		self._redis.lpush(queue_name, obj_bytes)

	def pop_queue_obj(self, queue_name):
		obj_bytes = self._redis.rpop(queue_name)
		if obj_bytes is None:
			return None
		return pickle.loads(obj_bytes)

	def publish(self, channel, message):
		self._redis.publish(channel, message)

	def save(self):
		self._redis.bgsave()