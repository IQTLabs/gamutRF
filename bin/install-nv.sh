#!/bin/bash
set -e
apt-get update && \
  apt-get install -y --no-install-recommends ca-certificates dirmngr gpg-agent gpg wget && \
  apt-key adv --fetch-keys "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/$(arch)/3bf863cc.pub" && \
  echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/$(arch)/ /" | tee /etc/apt/sources.list.d/nvidia.list && \
  apt-get update && \
  apt-get install -y --no-install-recommends libnvidia-gl-550
