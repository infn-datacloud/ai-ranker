ARG PYTHON_VERSION=3.11
ARG POETRY_VERSION=1.8.3

# Create requirements.txt from poetry dependencies
FROM ghcr.io/withlogicco/poetry:${POETRY_VERSION}-python-${PYTHON_VERSION}-slim AS requirements

ENV POETRY_VIRTUALENVS_CREATE=true
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

COPY ./pyproject.toml ./poetry.lock* ./README.md /usr/src/app/
COPY ./src/ /usr/src/app/src

RUN poetry install --without dev

# Stage used in production
FROM python:${PYTHON_VERSION}-slim AS production

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt file and install dependencies
COPY --from=requirements /usr/src/app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY ./src/ ./src
ENV PYTHONPATH=/app
