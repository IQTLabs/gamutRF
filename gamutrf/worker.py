import argparse
import json
import logging
import os
import queue
import socket
import struct
import sys
import threading
import time

import bjoern
import falcon
from falcon_cors import CORS

from gamutrf.__init__ import __version__
from gamutrf.birdseye_rssi import BirdsEyeRSSI
from gamutrf.birdseye_rssi import FLOAT_SIZE
from gamutrf.birdseye_rssi import MAX_RSSI
from gamutrf.birdseye_rssi import RSSI_UDP_ADDR
from gamutrf.birdseye_rssi import RSSI_UDP_PORT
from gamutrf.mqtt_reporter import MQTTReporter
from gamutrf.sdr_recorder import get_recorder
from gamutrf.sdr_recorder import RECORDER_MAP

WORKER_NAME = os.getenv("WORKER_NAME", socket.gethostbyname(socket.gethostname()))
ORCHESTRATOR = os.getenv("ORCHESTRATOR", "orchestrator")
ANTENNA = os.getenv("ANTENNA", "")


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loglevel",
        "-l",
        help="Set logging level",
        choices=["critical", "error", "warning", "info", "debug"],
        default="info",
    )
    parser.add_argument(
        "--antenna", "-a", help="Antenna make/model", type=str, default=ANTENNA
    )
    parser.add_argument(
        "--name", "-n", help="Name for the worker", type=str, default=WORKER_NAME
    )
    parser.add_argument(
        "--path",
        "-P",
        help="Path prefix for writing out samples to",
        type=str,
        default="/data/gamutrf",
    )
    parser.add_argument(
        "--port", "-p", help="Port to run the API webserver on", type=int, default=8000
    )
    parser.add_argument(
        "--rotate_secs",
        help="If > 0, rotate storage directories every N seconds",
        type=int,
        default=3600,
    )
    parser.add_argument(
        "--sdr",
        "-s",
        help=f"Specify SDR to record with {list(RECORDER_MAP.keys())} or file",
        type=str,
        default="ettus",
    )
    parser.add_argument(
        "--sdrargs",
        help=f"optional SDR arguments",
        type=str,
        default="",
    )
    parser.add_argument(
        "--freq_excluded",
        "-e",
        help='Freq range to exclude in MHz (e.g. "100-200")',
        action="append",
        default=[],
    )
    parser.add_argument("--gain", "-g", help="Gain in dB", default=30, type=int)
    parser.add_argument(
        "--mean_window", "-m", help="birdseye mean window size", default=128, type=int
    )
    parser.add_argument(
        "--rxb", help="Receive buffer size", default=int(1024 * 1024 * 10), type=int
    )
    parser.add_argument(
        "--qsize", help="Max request queue size", default=int(2), type=int
    )
    parser.add_argument(
        "--mqtt_server",
        help="MQTT server to report RSSI",
        default=ORCHESTRATOR,
        type=str,
    )
    parser.add_argument(
        "--gps_server",
        help="GPS Server to get lat,long, and heading",
        default=ORCHESTRATOR,
        type=str,
    )
    parser.add_argument(
        "--rssi_interval",
        help="rate limit in seconds for RSSI updates to MQTT",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--rssi_throttle",
        help="rate limit RSSI calculations to 1 in n",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--rssi_threshold", help="RSSI reporting threshold", default=-45, type=float
    )
    external_rssi_parser = parser.add_mutually_exclusive_group(required=False)
    external_rssi_parser.add_argument(
        "--rssi_external",
        dest="rssi_external",
        action="store_true",
        default=True,
        help="proxy external RSSI",
    )
    external_rssi_parser.add_argument(
        "--no-rssi_external",
        dest="rssi_external",
        action="store_false",
        help="do not use proxy external RSSI",
    )
    agc_parser = parser.add_mutually_exclusive_group(required=False)
    agc_parser.add_argument(
        "--agc", dest="agc", action="store_true", default=True, help="use AGC"
    )
    agc_parser.add_argument(
        "--no-agc", dest="agc", action="store_false", help="do not use AGC"
    )
    parser.add_argument(
        "--sigmf",
        dest="sigmf",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="add sigmf meta file",
    )
    parser.add_argument(
        "--use_external_gps",
        dest="use_external_gps",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use external Pixhawk/MAVLINK GPS",
    )
    parser.add_argument(
        "--use_external_heading",
        dest="use_external_heading",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use external (Pixhawk/MAVLINK) heading",
    )
    parser.add_argument(
        "--external_gps_server",
        dest="external_gps_server",
        default=ORCHESTRATOR,
        type=str,
        help="server to query for external GPS data",
    )
    parser.add_argument(
        "--external_gps_server_port",
        dest="external_gps_server_port",
        default="8888",
        type=str,
        help="server port to query for external GPS data",
    )

    return parser


class Endpoints:
    @staticmethod
    def on_get(_req, resp):
        endpoints = []
        for path in API.paths():
            endpoints.append(API.version() + path)

        resp.text = json.dumps(endpoints)
        resp.content_type = falcon.MEDIA_TEXT
        resp.status = falcon.HTTP_200


class Info:
    def __init__(self, arguments):
        self.arguments = arguments

    def on_get(self, _req, resp):
        resp.text = json.dumps(
            {
                "version": __version__,
                "sdr": self.arguments.sdr,
                "path_prefix": self.arguments.path,
                "freq_excluded": self.arguments.freq_excluded,
            }
        )
        resp.content_type = falcon.MEDIA_TEXT
        resp.status = falcon.HTTP_200


class Action:
    def __init__(self, arguments, q, sdr_recorder):
        self.arguments = arguments
        self.q = q
        self.sdr_recorder = sdr_recorder
        self.action = None

    def on_get(self, _req, resp, center_freq, sample_count, sample_rate):
        # TODO check if chosen SDR can do the supplied sample_count
        resp.content_type = falcon.MEDIA_JSON
        resp.status = falcon.HTTP_400

        status = None
        if self.q.full():
            status = "Request queue is full"
        else:
            status = self.sdr_recorder.validate_request(
                self.arguments.freq_excluded, center_freq, sample_count, sample_rate
            )

        if status is None:
            self.q.put(
                {
                    "action": self.action,
                    "center_freq": int(center_freq),
                    "sample_count": int(sample_count),
                    "sample_rate": int(sample_rate),
                }
            )
            status = "Requsted recording"
            resp.status = falcon.HTTP_200

        resp.text = json.dumps({"status": status})


class Record(Action):
    def __init__(self, arguments, q, sdr_recorder):
        super().__init__(arguments, q, sdr_recorder)
        self.action = "record"


class Rssi(Action):
    def __init__(self, arguments, q, sdr_recorder):
        super().__init__(arguments, q, sdr_recorder)
        self.action = "record"


class API:
    def __init__(self, arguments):
        self.arguments = arguments
        self.mqtt_reporter = MQTTReporter(
            name=self.arguments.name,
            mqtt_server=self.arguments.mqtt_server,
            gps_server=self.arguments.gps_server,
            compass=True,
            use_external_gps=self.arguments.use_external_gps,
            use_external_heading=self.arguments.use_external_heading,
            external_gps_server=self.arguments.external_gps_server,
            external_gps_server_port=self.arguments.external_gps_server_port,
        )
        self.q = queue.Queue(self.arguments.qsize)
        self.sdr_recorder = get_recorder(
            self.arguments.sdr, self.arguments.sdrargs, self.arguments.rotate_secs
        )
        self.start_time = time.time()
        cors = CORS(allow_all_origins=True)
        self.app = falcon.App(middleware=[cors.middleware])
        routes = self.routes()
        for route, handler in routes.items():
            self.app.add_route(self.version() + route, handler)

    def run_recorder(self, record_func):
        logging.info("run recorder")
        while True:
            logging.info("awaiting request")
            action_args = self.q.get()
            action = action_args["action"]
            if action == "record":
                self.serve_recording(record_func, action_args)
            elif action == "rssi":
                self.serve_rssi(action_args)
            else:
                logging.error("no such action: %s", action)

    def record(self, center_freq, sample_count, sample_rate=20e6):
        return self.sdr_recorder.run_recording(
            self.arguments.path,
            sample_rate,
            sample_count,
            center_freq,
            self.arguments.gain,
            self.arguments.agc,
            self.arguments.rxb,
            self.arguments.sigmf,
            self.arguments.sdr,
            self.arguments.antenna,
        )

    def serve_recording(self, record_func, record_args):
        logging.info(f"got a request: {record_args}")
        record_status = record_func(**record_args)
        if record_status == -1:
            # TODO this only kills the thread, not the main process
            return
        record_args.update(vars(self.arguments))
        self.mqtt_reporter.publish("gamutrf/record", record_args)
        self.mqtt_reporter.log(
            self.arguments.path, "record", self.start_time, record_args
        )

    def report_rssi(self, record_args, reported_rssi, reported_time):
        logging.info(f'reporting RSSI {reported_rssi} for {record_args["center_freq"]}')
        record_args.update({"rssi": reported_rssi, "time": reported_time})
        record_args.update(vars(self.arguments))
        self.mqtt_reporter.publish("gamutrf/rssi", record_args)
        self.mqtt_reporter.log(
            self.arguments.path, "rssi", self.start_time, record_args
        )

    def process_rssi(self, record_args, sock):
        last_rssi_time = 0
        duration = 0
        if record_args["sample_count"]:
            duration = float(record_args["sample_count"]) / float(
                record_args["sample_rate"]
            )
        start_time = time.time()
        while self.q.empty():
            rssi_raw, _ = sock.recvfrom(FLOAT_SIZE)
            rssi = struct.unpack("f", rssi_raw)[0]
            if rssi < self.arguments.rssi_threshold:
                continue
            if rssi > MAX_RSSI:
                continue
            now = time.time()
            if duration and now - start_time > duration:
                break
            now_diff = now - last_rssi_time
            if now_diff < self.arguments.rssi_interval:
                continue
            self.report_rssi(record_args, rssi, now)
            last_rssi_time = now

    def proxy_rssi(self, rssi_addr, record_args):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            # codeql[py/bind-socket-all-network-interfaces]
            sock.bind((rssi_addr, RSSI_UDP_PORT))  # nosec
            self.process_rssi(record_args, sock)

    def serve_rssi(self, record_args):
        logging.info(f"got request {record_args}")
        if self.arguments.rssi_external:
            logging.info("proxying external RSSI")
            # codeql[py/bind-socket-all-network-interfaces]
            self.proxy_rssi("0.0.0.0", record_args)  # nosec
        else:
            center_freq = int(record_args["center_freq"])
            try:
                rssi_server = BirdsEyeRSSI(
                    self.arguments,
                    record_args["sample_rate"],
                    center_freq,
                    agc=self.arguments.agc,
                    rssi_throttle=self.arguments.rssi_throttle,
                )
            except RuntimeError as err:
                logging.error("could not initialize RSSI server: %s", err)
                return
            rssi_server.start()
            logging.info(
                f"serving RSSI for {center_freq}Hz over threshold {self.arguments.rssi_threshold} with AGC {self.arguments.agc}"
            )
            self.proxy_rssi(RSSI_UDP_ADDR, record_args)
            logging.info("RSSI stream stopped")
            rssi_server.stop()
            rssi_server.wait()

    @staticmethod
    def paths():
        return [
            "",
            "/info",
            "/record/{center_freq}/{sample_count}/{sample_rate}",
            "/rssi/{center_freq}/{sample_count}/{sample_rate}",
        ]

    @staticmethod
    def version():
        return "/v1"

    def routes(self):
        p = self.paths()
        endpoints = Endpoints()
        info = Info(self.arguments)
        record = Record(self.arguments, self.q, self.sdr_recorder)
        rssi = Rssi(self.arguments, self.q, self.sdr_recorder)
        funcs = [endpoints, info, record, rssi]
        return dict(zip(p, funcs))

    def run(self):
        logging.info("starting recorder thread")
        recorder_thread = threading.Thread(
            target=self.run_recorder, args=(self.record,)
        )
        recorder_thread.start()

        logging.info("starting API thread")
        bjoern.run(self.app, "0.0.0.0", self.arguments.port)  # nosec
        recorder_thread.join()


def main():
    arguments = argument_parser().parse_args()
    level_int = {"CRITICAL": 50, "ERROR": 40, "WARNING": 30, "INFO": 20, "DEBUG": 10}
    level = level_int.get(arguments.loglevel.upper(), 0)
    logging.basicConfig(level=level, format="%(asctime)s %(message)s")
    try:
        app = API(arguments)
    except ValueError:
        sys.exit(1)
    app.run()
