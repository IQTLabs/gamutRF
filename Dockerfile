FROM ubuntu:20.04
COPY --from=iqtlabs/gamutrf-base:latest /usr/local /usr/local
COPY --from=iqtlabs/gamutrf-base:latest /usr/lib /usr/lib
COPY --from=iqtlabs/gamutrf-base:latest /usr/share/uhd/images /usr/share/uhd/images
LABEL maintainer="Charlie Lewis <clewis@iqt.org>"
ENV DEBIAN_FRONTEND noninteractive
ENV UHD_IMAGES_DIR /usr/share/uhd/images
# hadolint ignore=DL3008
RUN apt-get update && apt-get install --no-install-recommends -yq \
    git python3-numpy python3-pip && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY scan-requirements.txt /root/scan-requirements.txt
RUN pip3 install -r /root/scan-requirements.txt
COPY gamutrf/grscan.py /root/gamutrf/grscan.py
COPY gamutrf/utils.py /root/gamutrf/utils.py
COPY gamutrf/scan.py /root/scan.py
ENTRYPOINT ["/usr/bin/python3", "/root/scan.py"]
CMD ["--help"]
