# nosemgrep:github.workflows.config.dockerfile-source-not-pinned
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
    mesa-vulkan-drivers \
    python3 \
    python3-dev \
    libcairo2-dev \
    sox \
    wget \
    zstd && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.1.15 && \
    poetry config virtualenvs.create false
COPY . /gamutrf
WORKDIR /gamutrf
# TODO: https://github.com/python-poetry/poetry/issues/3591
# Install pandas via pip to get wheel. Disabling the new installer/configuring a wheel source does not work.
RUN rm -rf /usr/lib/python3/dist-packages/pycparser* && \
    poetry run pip install pandas==1.5.1 && \
    poetry install --no-interaction --no-ansi
# nosemgrep:github.workflows.config.missing-user
CMD ["gamutrf-scan", "--help"]
