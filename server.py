from flask import Flask, jsonify, request
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
from network.redis import RedisClient
from multiprocessing import Process
import pickle

app = Flask(__name__)
_redis = RedisClient()
# app.config["DEBUG"] = True

@app.route("/", methods=["GET"])
def home():
	return "<h1>Hello World</h1>"


@app.route('/upload', methods=["POST"])
def upload_files():
	files = request.files.to_dict()
	for _, file in files.items():
		img = Image.open(file).convert("RGB")
		_redis.push_queue_obj("queue:identifier", img)
	return jsonify("Uploaded!")

def _start():
	app.run()	

def start_process():
	process = Process(target=_start)
	process.daemon = True
	process.start()
	return process

