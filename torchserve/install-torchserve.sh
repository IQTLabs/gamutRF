#!/bin/bash

set -e
apt-get update && \
  apt-get install -y \
    git \
    python3-pip
pip config set global.no-cache-dir false && \
  git clone https://github.com/pytorch/serve -b v0.9.0 && \
  cd serve && \
  python3 ./ts_scripts/install_dependencies.py --environment prod $* && \
  pip3 install . && \
  pip3 install -r examples/object_detector/yolo/yolov8/requirements.txt && \
  cd .. && \
  rm -rf serve

