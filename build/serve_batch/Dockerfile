FROM python:3.8-slim

# ENVIRONMENT

ENV SERVICE_NAME=recommender
ENV API_VERSION = 1

WORKDIR /src

# Both files are explicitly required
COPY ./2_Serve_Batch_Inference/Pipfile  ./
COPY ./2_Serve_Batch_Inference/Pipfile.lock ./

RUN pip install pipenv
RUN apt-get update && \
    apt-get -y install gcc g++ python3-dev
RUN mkdir .venv
RUN pipenv install

RUN mkdir 0_Data

COPY ./2_Serve_Batch_Inference/. .

EXPOSE 5001
CMD ["pipenv", "run", "gunicorn", "--bind", "0.0.0.0:5001", "--workers=2", "--access-logfile=-", "api:app"]
