from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import numpy as np
from PIL import Image
from network.redis import RedisClient
from multiprocessing import Process
import pickle

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERES'] = 'Content-Type'
_redis = RedisClient()
# app.config["DEBUG"] = True

@app.route("/", methods=["GET"])
def home():
	return "<h1>Hello World</h1>"


@app.route('/upload', methods=["POST"])
@cross_origin()
def upload_files():
	files = request.files.to_dict()
	for _, file in files.items():
		img = Image.open(file).convert("RGB")
		_redis.push_queue_obj("queue:identifier", img)
	return jsonify("Uploaded!")

def _start():
	app.run(host="0.0.0.0", port="5000")	

def start_process():
	process = Process(target=_start)
	process.daemon = True
	process.start()
	return process

