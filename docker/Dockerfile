FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV GIT_PYTHON_REFRESH=quiet

WORKDIR /app

COPY pyproject.toml README.md ./
RUN mkdir -p src && touch src/__init__.py \
    && pip install --no-cache-dir "." \
    && rm -rf src

COPY . .

CMD ["bash"]
