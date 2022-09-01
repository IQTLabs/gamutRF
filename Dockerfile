FROM ubuntu:22.04
COPY --from=iqtlabs/gamutrf-base:latest /usr/local /usr/local
COPY --from=iqtlabs/gamutrf-base:latest /usr/lib /usr/lib
COPY --from=iqtlabs/gamutrf-base:latest /usr/share/uhd/images /usr/share/uhd/images
LABEL maintainer="Charlie Lewis <clewis@iqt.org>"
ENV DEBIAN_FRONTEND noninteractive
ENV UHD_IMAGES_DIR /usr/share/uhd/images
ENV PATH="${PATH}:/root/.local/bin"
RUN mkdir -p /data/gamutrf
RUN apt-get update && apt-get install --no-install-recommends -yq \
    ca-certificates \
    curl \
    ffmpeg \
    gcc \
    git \
    gnuplot \
    libev-dev \
    python3 \
    python3-dev \
    sox \
    wget \
    zstd && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://install.python-poetry.org | python3 - && \
  poetry config virtualenvs.create false
COPY . /gamutrf
WORKDIR /gamutrf
RUN poetry install --no-interaction --no-ansi
CMD ["gamutrf-scan", "--help"]
