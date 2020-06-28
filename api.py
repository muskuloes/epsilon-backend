import os
import uuid

from flask import Flask, request
from flask_restful import Resource, Api
from flask_socketio import SocketIO, emit
from flask_pymongo import PyMongo

from flask_cors import CORS

app = Flask(__name__)
app.config["MONGO_URI"] = os.getenv(
    "MONGO_URI", default="mongodb://localhost:27017/hca"
)
CORS(app)
mongo = PyMongo(app)
socketio = SocketIO(app, cors_allowed_origins="*")

api = Api(app)


class Test(Resource):
    def get(self):
        return {"Team": "Unicorn"}


class Upload(Resource):
    def get(self, filename):
        return mongo.send_file(filename)

    # TODO: return predicted images classes and attributes together with images url.
    def post(self, **kwargs):
        files = request.files
        data = []
        for _, file in files.items(multi=True):
            filename = "{}".format(uuid.uuid4())
            mongo.save_file(filename, file)
            # build image data from tensorflow model
            imageData = {"name": filename}
            id = mongo.db.imageData.insert_one(imageData).inserted_id
            data.append({"id": str(id), "name": filename})

        socketio.emit("updated files", data, broadcast=True)
        return {"data": data}


class Connect(Resource):
    def get(self):
        data = []
        for item in mongo.db.imageData.find():
            data.append({"id": str(item.get("_id")), "name": item.get("name")})
        return {"data": data}


api.add_resource(Connect, "/")
api.add_resource(Upload, "/upload", "/upload/<string:filename>")

api.add_resource(Test, "/test")

if __name__ == "__main__":
    socketio.run(app, debug=True)
