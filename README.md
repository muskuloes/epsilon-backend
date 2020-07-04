# Online Deployment

- The API is deployed on Heroku: https://hca-backend.herokuapp.com/test

# Development Setup

- Install [pipenv](https://pypi.org/project/pipenv/)
- Clone this repository
- Install project dependencies:

```sh
pipenv install
```

### Start the api:

- Using [docker-compose](https://docs.docker.com/compose/):

```sh
docker-compose up
```

- Try it out

```sh
curl localhost:5000/test

result:
{
  "Team": "Unicorn"
}
```
