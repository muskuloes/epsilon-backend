from flask import Flask, request
from flask_restful import Resource, Api
from flask_pymongo import PyMongo
import uuid

app = Flask(__name__)
app.config['MONGO_URI'] = 'mongodb://localhost:27017/hca'
mongo = PyMongo(app)

api = Api(app)


class HelloWorld(Resource):
    def get(self):
        return {'Team': 'Unicorn'}


class Upload(Resource):
    def get(self, filename):
        return mongo.send_file(filename)

    def post(self, **kwargs):
        file = request.files['file']
        file_extension = file.filename.split('.')[-1]
        filename = "{}.{}".format(uuid.uuid4(), file_extension)
        mongo.save_file(filename, request.files['file'])

        return {'filename': filename}


api.add_resource(HelloWorld, '/')
api.add_resource(Upload, '/upload', '/upload/<string:filename>')

if __name__ == '__main__':
    app.run(debug=True)
