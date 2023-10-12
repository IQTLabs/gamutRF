#!/bin/bash

set -e
TMPDIR=/tmp
sudo apt-get update && sudo apt-get install -y curl wget
sudo pip3 install torch-model-archiver
cd $TMPDIR
git clone https://github.com/pytorch/serve -b v0.8.2
cd serve/examples/object_detector/yolo/yolov8
# TODO: use gamutRF weights here.
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
torch-model-archiver --force --model-name yolov8n --version 1.0 --serialized-file yolov8n.pt --handler custom_handler.py
rm -rf model_store && mkdir model_store
mv yolov8n.mar model_store/
docker run -v $(pwd)/model_store:/model_store --net host -d iqtlabs/gamutrf-torchserve timeout 60s torchserve --start --model-store /model_store --ncs --foreground
sleep 5
curl -X POST "localhost:8081/models?model_name=yolov8n&url=yolov8n.mar&initial_workers=4&batch_size=2"
curl http://127.0.0.1:8080/predictions/yolov8n -T persons.jpg
