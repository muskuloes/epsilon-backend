# Development Setup

- Install [pipenv](https://pypi.org/project/pipenv/)
- Clone this repository
- Install project dependencies:

```sh
cd hca-backend
pipenv install
```

- Start the api:

```sh
pipenv shell
python api.py
```

- Start MongoDB

```
docker run --name mongodb -p 27017:27017 -d mongo
```
