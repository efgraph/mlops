FROM python:3.10.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.6.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PYTHONPATH="/app"

ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

RUN useradd -m -u 1000 appuser

RUN mkdir -p /app/models && chown -R appuser:appuser /app/models

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root --no-dev

COPY ag_news_classifier/ ./ag_news_classifier/
COPY conf/ ./conf/
COPY commands.py ./

RUN chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import transformers; import torch; import pytorch_lightning" || exit 1

ENTRYPOINT ["tail", "-f", "/dev/null"]
