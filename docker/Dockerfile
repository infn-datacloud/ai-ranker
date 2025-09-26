ARG PYTHON_VERSION=3.12
ARG POETRY_VERSION=2.1

# Create requirements.txt from poetry dependencies
FROM harbor.cloud.infn.it/datacloud-middleware/poetry:${POETRY_VERSION}-python${PYTHON_VERSION} AS base

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./pyproject.toml ./poetry.lock ./README.md /app/
COPY ./src /app/src

RUN poetry install --without dev

USER ${USERNAME}