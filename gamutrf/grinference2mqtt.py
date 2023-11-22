#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
import time
import numpy as np

try:
    from gnuradio import gr  # pytype: disable=import-error
    from gamutrf.mqtt_reporter import MQTTReporter
except ModuleNotFoundError as err:  # pragma: no cover
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme): %s"
        % err
    )
    sys.exit(1)


DELIM = "\n\n"


class inference2mqtt(gr.sync_block):
    def __init__(
        self,
        name,
        mqtt_server,
        gps_server,
        use_external_gps,
        use_external_heading,
        external_gps_server,
        external_gps_server_port,
        log_path,
    ):
        self.yaml_buffer = ""
        self.start_time = time.time()
        self.mqtt_reporter = MQTTReporter(
            name=name,
            mqtt_server=mqtt_server,
            gps_server=gps_server,
            compass=True,
            use_external_gps=use_external_gps,
            use_external_heading=use_external_heading,
            external_gps_server=external_gps_server,
            external_gps_server_port=external_gps_server_port,
        )
        self.log_path = log_path

        gr.sync_block.__init__(
            self,
            name="inference2mqtt",
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

    def process_item(self, item):
        self.mqtt_reporter.publish("gamutrf/inference", item)
        self.mqtt_reporter.log(self.log_path, "inference", self.start_time, item)
        return
