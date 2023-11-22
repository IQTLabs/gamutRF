# nosemgrep:github.workflows.config.dockerfile-source-not-pinned
FROM python:3.12-slim@sha256:32477c7ecb3b18a47cff8eb5b8149bc21c858998b0ae2bb36f4ddb43cc79d240

WORKDIR /app
ENV PYTHONPATH=${PYTHONPATH}:${PWD}

RUN pip3 install poetry==1.1.5
COPY pyproject.toml .
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

COPY mavlink-api.py .

# nosemgrep:github.workflows.config.missing-user
CMD ["python3", "mavlink-api.py"]