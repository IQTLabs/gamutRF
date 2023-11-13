# nosemgrep:github.workflows.config.dockerfile-source-not-pinned
FROM python:3.11-slim@sha256:f89d4d260b6a5caa6aa8e0e14b162deb76862890c91779c31f762b22e72a6cef

WORKDIR /app
ENV PYTHONPATH=${PYTHONPATH}:${PWD}

RUN pip3 install poetry==1.1.5
COPY pyproject.toml .
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

COPY mavlink-api.py .

# nosemgrep:github.workflows.config.missing-user
CMD ["python3", "mavlink-api.py"]