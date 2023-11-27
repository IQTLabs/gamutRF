#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import queue
import sys
import threading
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
        compass,
        gps_server,
        use_external_gps,
        use_external_heading,
        external_gps_server,
        external_gps_server_port,
        log_path,
    ):
        self.yaml_buffer = ""
        self.mqtt_reporter = None
        self.q = queue.Queue()
        self.mqtt_reporter_thread = threading.Thread(
            target=self.reporter_thread,
            args=(
                name,
                mqtt_server,
                gps_server,
                compass,
                use_external_gps,
                use_external_heading,
                external_gps_server,
                external_gps_server_port,
                log_path,
            ),
        )
        self.mqtt_reporter_thread.start()

        gr.sync_block.__init__(
            self,
            name="inference2mqtt",
            in_sig=[np.ubyte],
            out_sig=None,
        )

    def reporter_thread(
        self,
        name,
        mqtt_server,
        gps_server,
        compass,
        use_external_gps,
        use_external_heading,
        external_gps_server,
        external_gps_server_port,
        log_path,
    ):
        start_time = time.time()
        mqtt_reporter = MQTTReporter(
            name=name,
            mqtt_server=mqtt_server,
            gps_server=gps_server,
            compass=compass,
            use_external_gps=use_external_gps,
            use_external_heading=use_external_heading,
            external_gps_server=external_gps_server,
            external_gps_server_port=external_gps_server_port,
        )
        while True:
            item = self.q.get()
            mqtt_reporter.publish("gamutrf/inference", item)
            mqtt_reporter.log(log_path, "inference", start_time, item)
            self.q.task_done()

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
        self.q.put(item)
