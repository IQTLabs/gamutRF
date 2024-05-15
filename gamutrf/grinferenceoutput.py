#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import logging
import queue
import sys
import threading
import time
import pmt
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


class inferenceoutput(gr.basic_block):
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
    ):
        self.q = queue.Queue()
        self.running = True
        self.serialno = 0
        self.reporter_thread = threading.Thread(
            target=self.run_reporter_thread,
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
        gr.basic_block.__init__(
            self,
            name="inferenceoutput",
            in_sig=None,
            out_sig=None,
        )
        self.message_port_register_in(pmt.intern("inference"))
        self.set_msg_handler(pmt.intern("inference"), self.receive_pdu)

    def receive_pdu(self, pdu):
        self.q.put(json.loads(bytes(pmt.to_python(pmt.cdr(pdu))).decode("utf8")))

    def stop(self):
        self.running = False
        self.reporter_thread.join()

    def run_reporter_thread(
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
            logging.info("inference output %u: %s", self.serialno, item)
            self.serialno += 1
            if zmq_pub is not None:
                zmq_pub.send_string(json.dumps(item) + DELIM, flags=zmq.NOBLOCK)
            if mqtt_reporter is not None:
                mqtt_reporter.publish("gamutrf/inference", item)
                mqtt_reporter.log(log_path, "inference", start_time, item)
            self.q.task_done()
