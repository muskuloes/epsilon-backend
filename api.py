import os
import uuid

from flask import Flask, request
from flask_restful import Resource, Api
from flask_pymongo import PyMongo
from flask_cors import CORS

app = Flask(__name__)
app.config["MONGO_URI"] = os.getenv(
    "MONGO_URI", default="mongodb://localhost:27017/hca"
)
CORS(app)
mongo = PyMongo(app)

api = Api(app)


class HelloWorld(Resource):
    def get(self):
        return {"Team": "Unicorn"}


class Upload(Resource):
    def get(self, filename):
        return mongo.send_file(filename)

    def post(self, **kwargs):
        files = request.files
        filenames = []
        for param, file in files.items(multi=True):
            file_extension = file.filename.split(".")[-1]
            filename = "{}.{}".format(uuid.uuid4(), file_extension)
            filenames.append(filename)
            mongo.save_file(filename, file)

        return {"filenames": filenames}


api.add_resource(HelloWorld, "/")
api.add_resource(Upload, "/upload", "/upload/<string:filename>")

if __name__ == "__main__":
    app.run(debug=True)
