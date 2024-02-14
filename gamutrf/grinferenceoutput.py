#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import logging
import queue
import sys
import threading
import time
import numpy as np
import zmq

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


class jsonmixer:
    def __init__(self, inputs):
        self.inputs = inputs
        self.json_buffer = {}
        for i in range(self.inputs):
            self.json_buffer[i] = ""

    def mix(self, input_items):
        items = []
        n = 0
        for i, input_item in enumerate(input_items):
            raw_input_item = input_item.tobytes().decode("utf8")
            n += len(raw_input_item)
            self.json_buffer[i] += raw_input_item
            while True:
                delim_pos = self.json_buffer[i].find(DELIM)
                if delim_pos == -1:
                    break
                raw_item = self.json_buffer[i][:delim_pos]
                self.json_buffer[i] = self.json_buffer[i][delim_pos + len(DELIM) :]
                try:
                    item = json.loads(raw_item)
                    items.append(item)
                except json.JSONDecodeError as e:
                    logging.error("cannot decode %s: %s", raw_item, e)
        return (n, items)


class inferenceoutput(gr.sync_block):
    def __init__(
        self,
        name,
        zmq_addr,
        mqtt_server,
        compass,
        gps_server,
        use_external_gps,
        use_external_heading,
        external_gps_server,
        external_gps_server_port,
        log_path,
        inputs,
    ):
        self.mixer = jsonmixer(inputs)
        self.q = queue.Queue()
        self.running = True
        self.reporter_thread = threading.Thread(
            target=self.reporter_thread,
            args=(
                name,
                zmq_addr,
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
        self.reporter_thread.start()
        gr.sync_block.__init__(
            self,
            name="inferenceoutput",
            in_sig=([np.ubyte] * inputs),
            out_sig=None,
        )

    def stop(self):
        self.running = False
        self.reporter_thread.join()

    def reporter_thread(
        self,
        name,
        zmq_addr,
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
        zmq_context = None
        zmq_pub = None
        if zmq_addr:
            zmq_context = zmq.Context()
            zmq_pub = zmq_context.socket(zmq.PUB)
            zmq_pub.setsockopt(zmq.SNDHWM, 100)
            zmq_pub.setsockopt(zmq.SNDBUF, 65536)
            zmq_pub.bind(zmq_addr)
        mqtt_reporter = None
        if mqtt_server:
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
        while self.running:
            try:
                item = self.q.get(block=True, timeout=1)
            except queue.Empty:
                continue
            logging.info("inference output %s", item)
            if zmq_pub is not None:
                zmq_pub.send_string(json.dumps(item) + DELIM, flags=zmq.NOBLOCK)
            if mqtt_reporter is not None:
                mqtt_reporter.publish("gamutrf/inference", item)
                mqtt_reporter.log(log_path, "inference", start_time, item)
            self.q.task_done()

    def work(self, input_items, output_items):
        n, items = self.mixer.mix(input_items)
        for item in items:
            self.q.put(item)
        return n
