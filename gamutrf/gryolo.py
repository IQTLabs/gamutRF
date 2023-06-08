#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pmt
import sys
from pathlib import Path

try:
    from gnuradio import gr  # pytype: disable=import-error
except ModuleNotFoundError as err:  # pragma: no cover
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme): %s"
        % err
    )
    sys.exit(1)


class terminal_sink(gr.sync_block):
    # Prints output layer of classifier to terminal for troubleshooting
    def __init__(self, input_vlen, batch_size):
        self.input_vlen = input_vlen
        self.batch_size = batch_size
        gr.sync_block.__init__(
            self,
            name="terminal_sink",
            in_sig=[(np.float32, self.input_vlen)],
            out_sig=None,
        )
        self.batch_ctr = 0

    def work(self, input_items, output_items):
        in0 = input_items[0]
        batch = in0.reshape(self.batch_size, -1)
        self.batch_ctr += 1
        return len(input_items[0])


class yolo_bbox(gr.sync_block):
    def __init__(
        self,
        image_shape,
        prediction_shape,
        batch_size,
        sample_rate,
        output_dir,
        confidence_threshold=0.5,
        nms_threshold=0.5,
    ):
        self.image_shape = image_shape
        self.image_vlen = np.prod(image_shape)
        self.prediction_shape = prediction_shape
        self.prediction_vlen = np.prod(prediction_shape)
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        gr.sync_block.__init__(
            self,
            name="yolo_bbox",
            in_sig=[(np.float32, self.image_vlen), (np.float32, self.prediction_vlen)],
            out_sig=None,
        )

    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        # label = f'{CLASSES[class_id]} ({confidence:.2f})'
        # label = f'{class_id} ({confidence:.2f})'
        label = f"{class_id}"
        color = (255, 255, 255)  # self.colors[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(
            img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    def work(self, input_items, output_items):
        rx_times = [
            sum(pmt.to_python(rx_time_pmt.value))
            for rx_time_pmt in self.get_tags_in_window(
                0, 0, len(input_items[0]), pmt.to_pmt("rx_time")
            )
        ]
        rx_freqs = [
            pmt.to_python(rx_freq_pmt.value)
            for rx_freq_pmt in self.get_tags_in_window(
                0, 0, len(input_items[0]), pmt.to_pmt("rx_freq")
            )
        ]

        image = input_items[0][0]
        image = image.reshape(self.image_shape)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        original_image = image
        [height, width, _] = original_image.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image
        scale = length / 640

        prediction = input_items[1][0]
        prediction = prediction.reshape(self.prediction_shape)
        prediction = np.array([cv2.transpose(prediction[0])])
        rows = prediction.shape[1]

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = prediction[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(
                classes_scores
            )
            if maxScore >= self.confidence_threshold:
                box = [
                    prediction[0][i][0] - (0.5 * prediction[0][i][2]),
                    prediction[0][i][1] - (0.5 * prediction[0][i][3]),
                    prediction[0][i][2],
                    prediction[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        result_boxes = cv2.dnn.NMSBoxes(
            boxes, scores, self.confidence_threshold, self.nms_threshold, 0.5, 200
        )

        detections = []
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                "class_id": class_ids[index],
                #'class_name': CLASSES[class_ids[index]],
                "confidence": scores[index],
                "box": box,
                "scale": scale,
            }
            detections.append(detection)
            self.draw_bounding_box(
                original_image,
                class_ids[index],
                scores[index],
                round(box[0] * scale),
                round(box[1] * scale),
                round((box[0] + box[2]) * scale),
                round((box[1] + box[3]) * scale),
            )

        Path(self.output_dir, "predictions").mkdir(parents=True, exist_ok=True)
        filename = str(
            Path(
                self.output_dir,
                "predictions",
                f"prediction_{rx_times[-1]:.3f}_{rx_freqs[-1]:.0f}Hz_{self.sample_rate:.0f}sps.png",
            )
        )
        cv2.imwrite(filename, original_image)

        return 1  # len(input_items[0])
