#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np

try:
    from gnuradio import gr  # pytype: disable=import-error
except ModuleNotFoundError as err:  # pragma: no cover
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme): %s"
        % err
    )
    sys.exit(1)


DELIM = "\n\n"


class yolo_bbox(gr.sync_block):
    def __init__(
        self,
        output_dir,
        confidence_threshold=0.5,
        nms_threshold=0.5,
    ):
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.yaml_buffer = ""

        gr.sync_block.__init__(
            self,
            name="yolo_bbox",
            in_sig=[np.ubyte],
            out_sig=None,
        )

    def work(self, input_items, output_items):
        n = 0
        for input_item in input_items:
            raw_input_item = input_item.tobytes().decode("utf8")
            n += len(raw_input_item)
            self.yaml_buffer += raw_input_item
        while True:
            delim_pos = self.yaml_buffer.find(DELIM)
            if delim_pos == -1:
                break
            raw_item = self.yaml_buffer[:delim_pos]
            item = json.loads(raw_item)
            self.yaml_buffer = self.yaml_buffer[delim_pos + len(DELIM) :]
            self.process_item(item)
        return n

    def draw_bounding_box(self, img, name, confidence, x, y, x_plus_w, y_plus_h):
        label = f"{name}: {confidence}"
        color = (255, 255, 255)
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(
            img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    def process_item(self, item):
        predictions = item.get("predictions", None)
        if not predictions:
            return

        boxes = []
        scores = []
        detections = []

        try:
            for name, prediction_data in predictions.items():
                for prediction in prediction_data:
                    conf = prediction["conf"]
                    if conf < self.confidence_threshold:
                        continue
                    xywh = prediction["xywh"]
                    box = [
                        xywh[0] - (0.5 * xywh[2]),
                        xywh[1] - (0.5 * xywh[3]),
                        xywh[2],
                        xywh[3],
                    ]
                    detections.append({"box": box, "score": conf, "name": name})
                    boxes.append(box)
                    scores.append(conf)
        except TypeError as e:
            print(f"invalid predictions from torchserve: {e}, {predictions}")
            return

        if not detections:
            return

        original_image = cv2.imread(item["image_path"])
        result_boxes = cv2.dnn.NMSBoxes(
            boxes, scores, self.confidence_threshold, self.nms_threshold, 0.5, 200
        )

        # TODO: output to ZMQ
        for detection in detections:
            self.draw_bounding_box(
                original_image,
                detection["name"],
                detection["score"],
                round(box[0]),
                round(box[1]),
                round(box[0] + box[2]),
                round(box[1] + box[3]),
            )

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        filename = str(
            Path(
                self.output_dir,
                "_".join(["prediction", os.path.basename(item["image_path"])]),
            )
        )
        cv2.imwrite(filename, original_image)
