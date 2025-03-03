ARG PYTHON_VERSION=3.10
ARG POETRY_VERSION=1.8.3

# Create requirements.txt from poetry dependencies
FROM ghcr.io/withlogicco/poetry:${POETRY_VERSION}-python-${PYTHON_VERSION}-slim AS requirements

WORKDIR /tmp

COPY ./pyproject.toml ./poetry.lock* /tmp/

RUN poetry export \
    -f requirements.txt \
    --output requirements.txt \
    --without-hashes \
    --without dev

# Stage used in production
FROM python:${PYTHON_VERSION}-slim AS production

WORKDIR /app

# Copy requirements.txt file and install dependencies
COPY --from=requirements /tmp/requirements.txt /app/requirements.txt

RUN pip install --user --upgrade pip && \
    pip install --user --no-cache-dir -r /app/requirements.txt

COPY src/ .
