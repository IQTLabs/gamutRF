# nosemgrep:github.workflows.config.dockerfile-source-not-pinned
FROM python:3.12-slim@sha256:80571b64ab7b94950d49d413f074e1932b65f6f75e0c34747b40ea41889a2ca9

WORKDIR /app
ENV PYTHONPATH=${PYTHONPATH}:${PWD}

RUN pip3 install poetry==1.1.5
COPY pyproject.toml .
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

COPY mavlink-api.py .

# nosemgrep:github.workflows.config.missing-user
CMD ["python3", "mavlink-api.py"]