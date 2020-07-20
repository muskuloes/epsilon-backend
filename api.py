from bson.objectid import ObjectId
from bson.json_util import dumps
import os
import requests
import uuid

from flask import Flask, request
from flask_restful import Resource, Api
from flask_socketio import SocketIO, emit
from flask_pymongo import PyMongo
from flask_cors import CORS

from model import detect
from celery_init import make_celery


api_url = os.getenv("API_URL", default="http://api:5000")
mongo_uri = os.getenv("MONGODB_URI", default="mongodb://mongodb:27017/epsilon")
celery_broker_url = os.getenv("CLOUDAMQP_URL", default="amqp://rabbitmq:5672")
celery_result_backend = mongo_uri
port = os.getenv("PORT", default=5000)

app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL=celery_broker_url, CELERY_RESULT_BACKEND=celery_result_backend
)
celery = make_celery(app)
app.config["MONGO_URI"] = mongo_uri
CORS(app)
mongo = PyMongo(app)
socketio = SocketIO(app, cors_allowed_origins="*")
api = Api(app)


@celery.task(name="predict")
def predict(id, filename):
    file = requests.get("{}/upload/{}".format(api_url, filename))
    preds = detect(file.content)
    mongo.db.imageData.update_one(
        {"_id": ObjectId(id)}, {"$set": {"preds": preds}}, upsert=False
    )


class Test(Resource):
    def get(self):
        return {"Team": "Unicorn"}


class Search(Resource):
    def get(self, term):
        data = mongo.db.imageData.find({"preds.class_ids": term})
        return {"data": dumps(data)}


class Upload(Resource):
    def get(self, filename):
        return mongo.send_file(filename)

    # TODO: return predicted images classes and attributes together with images url.
    def post(self, **kwargs):
        files = request.files
        data = []
        for _, file in files.items(multi=True):
            file_extension = file.filename.split(".")[-1]
            filename = "{}.{}".format(uuid.uuid4(), file_extension)
            mongo.save_file(filename, file)
            imageData = {"name": filename, "preds": {}}
            id = mongo.db.imageData.insert_one(imageData).inserted_id
            # predict from tensorflow model
            predict.apply_async(queue="predict", args=(str(id), filename))
            data.append({"id": str(id), "name": filename})

        socketio.emit("updated files", data, broadcast=True)
        return {"data": data}


class Prediction(Resource):
    def get(self, filename):
        while True:
            data = mongo.db.imageData.find_one({"name": filename}, {"_id": 0})
            if data and data["preds"]:
                return {"data": dumps(data)}


class Connect(Resource):
    def get(self):
        data = []
        for item in mongo.db.imageData.find():
            data.append({"id": str(item.get("_id")), "name": item.get("name")})
        return {"data": data}


api.add_resource(Connect, "/")
api.add_resource(Prediction, "/predictions/<string:filename>")
api.add_resource(Upload, "/upload", "/upload/<string:filename>")
api.add_resource(Search, "/search/<string:term>")
api.add_resource(Test, "/test")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
