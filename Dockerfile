# nosemgrep:github.workflows.config.dockerfile-source-not-pinned
FROM ubuntu:22.04 as installer
COPY --from=iqtlabs/gamutrf-base:latest /usr/local /usr/local
ENV DEBIAN_FRONTEND noninteractive
ENV PATH="${PATH}:/root/.local/bin"
WORKDIR /gamutrf
COPY gamutrf gamutrf/
COPY bin bin/
COPY templates templates/
COPY poetry.lock pyproject.toml README.md /gamutrf/
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# TODO: https://github.com/python-poetry/poetry/issues/3591
# Install pandas via pip to get wheel. Disabling the new installer/configuring a wheel source does not work.
RUN apt-get update && apt-get install --no-install-recommends -y -q \
    ca-certificates \
    curl \
    gcc \
    git \
    libcairo2-dev \
    libev-dev \
    python3 \
    python3-dev \
    python3-pip && \
    curl -sSL https://install.python-poetry.org | python3 - --version 1.4.2 && \
    poetry config virtualenvs.create false && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    poetry run pip install --no-cache-dir pandas==$(grep pandas pyproject.toml | grep -Eo "[0-9\.]+") && \
    poetry install --no-interaction --no-dev --no-ansi

# nosemgrep:github.workflows.config.dockerfile-source-not-pinned
FROM ubuntu:22.04
COPY --from=installer /usr/local /usr/local
COPY --from=installer /gamutrf /gamutrf
COPY --from=installer /root/.local /root/.local
COPY --from=iqtlabs/gamutrf-base:latest /usr/share/uhd/images /usr/share/uhd/images
LABEL maintainer="Charlie Lewis <clewis@iqt.org>"
ENV DEBIAN_FRONTEND noninteractive
ENV UHD_IMAGES_DIR /usr/share/uhd/images
ENV PATH="${PATH}:/root/.local/bin"
RUN mkdir -p /data/gamutrf
RUN apt-get update && apt-get install --no-install-recommends -y -q \
    ca-certificates \
    libblas3 \
    libboost-iostreams1.74.0 \
    libboost-program-options1.74.0 \
    libboost-thread1.74.0 \
    libcairo2 \
    libev4 \
    libfftw3-3 \
    libgl1 \
    libglib2.0-0 \
    liblapack3 \
    libopencv-core4.5d \
    libopencv-imgcodecs4.5d \
    libopencv-imgproc4.5d \
    libspdlog1 \
    libuhd4.1.0 \
    libunwind8 \
    libzmq5 \
    mesa-vulkan-drivers \
    python3 \
    python3-zmq \
    sox \
    sudo \
    wget \
    zstd && \
    apt-get -y -q clean && rm -rf /var/lib/apt/lists/*
WORKDIR /gamutrf
RUN echo "$(find /gamutrf/gamutrf -type f -name \*py -print)"|xargs grep -Eh "^(import|from)\s"|grep -Ev "gamutrf"|sort|uniq|python3
RUN ldd /usr/local/bin/uhd_sample_recorder
# nosemgrep:github.workflows.config.missing-user
CMD ["gamutrf-scan", "--help"]
