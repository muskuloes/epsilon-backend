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


class HelloWorld(Resource):
    def get(self):
        return {"Team": "Unicorn"}


class Upload(Resource):
    def get(self, filename):
        return mongo.send_file(filename)

    # TODO: return predicted images classes and attributes together with images url.
    def post(self, **kwargs):
        files = request.files
        filenames = []
        for param, file in files.items(multi=True):
            file_extension = file.filename.split(".")[-1]
            filename = "{}.{}".format(uuid.uuid4(), file_extension)
            filenames.append(filename)
            mongo.save_file(filename, file)

        socketio.emit("updated files", filenames, broadcast=True)
        return {"filenames": filenames}


api.add_resource(HelloWorld, "/")
api.add_resource(Upload, "/upload", "/upload/<string:filename>")

if __name__ == "__main__":
    socketio.run(app, debug=True)
