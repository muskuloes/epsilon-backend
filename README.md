# Development Setup

- Install [pipenv](https://pypi.org/project/pipenv/)
- Clone this repository
- Install project dependencies:

```sh
cd hca-backend
pipenv install
```

- Start MongoDB

```
docker run --name mongodb -p 27017:27017 -d mongo
```

### Start the api:

- In the shell
```sh
pipenv shell
python api.py
```
- Using Docker: You'll have to create a docker network and start mongo db within this network as shown below
```sh
docker build -t api .
docker network create unicorn
docker run --name mongodb --network unicorn -p 27017:27017 -d mongo
docker run --name api --network unicorn --env MONGO_URI=mongodb://mongodb:27017/hca -p 5000:5000 api:latest
```

- Test
```sh
curl localhost:5000/test

result:
{
  "Team": "Unicorn"
}
```
