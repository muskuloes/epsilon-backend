FROM python:3.7

COPY Pipfile Pipfile.lock /app/

RUN pip install --upgrade pip
RUN pip install pipenv

COPY . /app
WORKDIR /app

RUN pipenv install --system --deploy

EXPOSE 5000

CMD ["python", "api.py"]
