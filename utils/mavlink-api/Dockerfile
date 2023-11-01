FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=${PYTHONPATH}:${PWD}

RUN pip3 install poetry
COPY pyproject.toml .
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

COPY mavlink-api.py .

ENTRYPOINT ["python3", "-u", "mavlink-api.py"]