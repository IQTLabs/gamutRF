# nosemgrep:github.workflows.config.dockerfile-source-not-pinned
FROM ubuntu:24.04 AS installer
ARG POETRY_CACHE
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="${PATH}:/root/.local/bin"
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /root/.config/pip
COPY pip.conf pip.conf
WORKDIR /root
COPY renovate.json /root/
RUN apt-get update && apt-get install --no-install-recommends -y -q \
    ca-certificates \
    curl \
    gcc \
    git \
    libev-dev \
    jq \
    python3 \
    python3-dev \
    python3-pip && \
    curl -sSL https://install.python-poetry.org | python3 - --version "$(jq -r .constraints.poetry /root/renovate.json)" && \
    poetry config virtualenvs.create false
COPY --from=iqtlabs/gamutrf-base:latest /usr/local /usr/local
WORKDIR /gamutrf
COPY poetry.lock pyproject.toml README.md /gamutrf/
# dependency install is cached for faster rebuild, if only gamutrf source changed.
RUN if [ "${POETRY_CACHE}" != "" ] ; then echo using cache "${POETRY_CACHE}" ; poetry source add --priority=default local "${POETRY_CACHE}" ; fi
# TODO: handle caching
RUN for i in bjoern falcon-cors gpsd-py3 ; do poetry run pip install --no-cache-dir "$i"=="$(grep $i pyproject.toml | grep -Eo '\"[0-9\.]+' | sed 's/\"//g')" || exit 1 ; done
RUN poetry install --no-interaction --no-ansi --no-dev --no-root
COPY gamutrf gamutrf/
COPY bin bin/
RUN poetry install --no-interaction --no-ansi --no-dev

# nosemgrep:github.workflows.config.dockerfile-source-not-pinned
FROM ubuntu:24.04
ARG POETRY_CACHE
LABEL maintainer="Charlie Lewis <clewis@iqt.org>"
ENV DEBIAN_FRONTEND=noninteractive
ENV UHD_IMAGES_DIR=/usr/share/uhd/images
ENV PATH="${PATH}:/root/.local/bin"
WORKDIR /root/.config/pip
COPY pip.conf pip.conf
WORKDIR /root
COPY bin/install-nv.sh /root
RUN mkdir -p /data/gamutrf
# install nvidia's vulkan support if x86.
# hadolint ignore=DL3008
RUN if [ "$(arch)" = "x86_64" ] ; then /root/install-nv.sh ; fi && \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libblas3 \
        libboost-iostreams1.83.0 \
        libboost-program-options1.83.0 \
        libboost-thread1.83.0 \
        libev4 \
        libfftw3-bin \
        libgl1 \
        libglib2.0-0 \
        liblapack3 \
        libopencv-core406t64 \
        libopencv-imgcodecs406t64 \
        libopencv-imgproc406t64 \
        librtlsdr2 \
        libspdlog1.12 \
        libuhd4.6.0 \
        libunwind8 \
        libvulkan1 \
        libzmq5 \
        mesa-vulkan-drivers \
        python3 \
        python3-pytest \
        python3-zmq \
        uhd-host \
        wget \
        zstd && \
    apt-get -y -q clean && rm -rf /var/lib/apt/lists/*
WORKDIR /
COPY --from=iqtlabs/gnuradio:3.10.11.0 /usr/share/uhd/images /usr/share/uhd/images
COPY --from=installer /usr/local /usr/local
COPY --from=installer /gamutrf /gamutrf
COPY tests /tests
COPY --from=installer /root/.local /root/.local
RUN ldconfig -v
RUN pytest tests
WORKDIR /gamutrf
RUN echo "$(find /gamutrf/gamutrf -type f -name \*py -print)"|xargs grep -Eh "^(import|from)\s"|grep -Ev "gamutrf"|sort|uniq|python3
# nosemgrep:github.workflows.config.missing-user
CMD ["gamutrf-scan", "--help"]
