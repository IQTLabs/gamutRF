FROM ubuntu:20.04
LABEL maintainer="Charlie Lewis <clewis@iqt.org>"
ENV DEBIAN_FRONTEND noninteractive
ENV UHD_IMAGES_DIR /usr/share/uhd/images
COPY --from=iqtlabs/gamutrf-builder:latest /usr/local /usr/local
COPY --from=iqtlabs/gamutrf-builder:latest /usr/lib/*-linux-gnu /usr/lib/
RUN apt-get update && apt-get install --no-install-recommends -yq \
    python3-pip

COPY gamutrf/scan.py /root/scan.py

ENTRYPOINT ["/usr/bin/python3", "/root/scan.py"]
CMD ["--help"]
