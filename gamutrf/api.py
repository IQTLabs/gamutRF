import argparse
import json
import logging
import os
import queue
import socket
import struct
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

WORKER_NAME = os.getenv(
    'WORKER_NAME', socket.gethostbyname(socket.gethostname()))
ORCHESTRATOR = os.getenv('ORCHESTRATOR', 'orchestrator')
ANTENNA = os.getenv('ANTENNA', '')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--loglevel', '-l', help='Set logging level',
    choices=['critical', 'error', 'warning', 'info', 'debug'], default='info')
parser.add_argument(
    '--antenna', '-a', help='Antenna make/model',
    type=str, default=ANTENNA)
parser.add_argument(
    '--name', '-n', help='Name for the worker',
    type=str, default=WORKER_NAME)
parser.add_argument(
    '--path', '-P', help='Path prefix for writing out samples to',
    type=str, default='/data/gamutrf')
parser.add_argument(
    '--port', '-p', help='Port to run the API webserver on', type=int,
    default=8000)
parser.add_argument(
    '--sdr', '-s', help=f'Specify SDR to record with {list(RECORDER_MAP.keys())} or file', type=str,
    default='ettus')
parser.add_argument(
    '--freq_excluded', '-e', help='Freq range to exclude in MHz (e.g. "100-200")',
    action='append', default=[])
parser.add_argument(
    '--gain', '-g', help='Gain in dB',
    default=30, type=int)
parser.add_argument(
    '--mean_window', '-m', help='birdseye mean window size',
    default=128, type=int)
parser.add_argument(
    '--rxb', help='Receive buffer size',
    default=int(1024 * 1024 * 10), type=int)
parser.add_argument(
    '--qsize', help='Max request queue size',
    default=int(2), type=int)
parser.add_argument(
    '--mqtt_server', help='MQTT server to report RSSI',
    default=ORCHESTRATOR, type=str)
parser.add_argument(
    '--rssi_interval', help='rate limit in seconds for RSSI updates to MQTT',
    default=1.0, type=float)
parser.add_argument(
    '--rssi_throttle', help='rate limit RSSI calculations to 1 in n',
    default=10, type=int)
parser.add_argument(
    '--rssi_threshold', help='RSSI reporting threshold',
    default=-45, type=float)
arg_parser = parser.add_mutually_exclusive_group(required=False)
arg_parser.add_argument(
    '--agc', dest='agc', action='store_true', default=True, help='use AGC')
arg_parser.add_argument(
    '--no-agc', dest='agc', action='store_false', help='do not use AGC')
sigmf_parser = parser.add_mutually_exclusive_group(required=False)
sigmf_parser.add_argument(
    '--sigmf', dest='sigmf', action='store_true', help='add sigmf meta file')
sigmf_parser.add_argument(
    '--no-sigmf', dest='sigmf', action='store_false', help='do not add sigmf meta file')
rssi_parser = parser.add_mutually_exclusive_group(required=False)
rssi_parser.add_argument(
    '--rssi', dest='enable_rssi', action='store_true', help='get RSSI values')
rssi_parser.add_argument(
    '--no-rssi', dest='enable_rssi', action='store_false', help='do not get RSSI values')

arguments = parser.parse_args()
q = queue.Queue(arguments.qsize)

sdr_recorder = get_recorder(arguments.sdr)

level_int = {'CRITICAL': 50, 'ERROR': 40, 'WARNING': 30, 'INFO': 20,
             'DEBUG': 10}
level = level_int.get(arguments.loglevel.upper(), 0)
logging.basicConfig(level=level, format='%(asctime)s %(message)s')


class Endpoints:

    @staticmethod
    def on_get(_req, resp):
        endpoints = []
        for path in API.paths():
            endpoints.append(API.version()+path)

        resp.text = json.dumps(endpoints)
        resp.content_type = falcon.MEDIA_TEXT
        resp.status = falcon.HTTP_200


class Info:

    @staticmethod
    def on_get(_req, resp):
        resp.text = json.dumps(
            {'version': __version__, 'sdr': arguments.sdr,
             'path_prefix': arguments.path, 'freq_excluded': arguments.freq_excluded})
        resp.content_type = falcon.MEDIA_TEXT
        resp.status = falcon.HTTP_200


class Record:

    @staticmethod
    def on_get(_req, resp, center_freq, sample_count, sample_rate):
        # TODO check if chosen SDR can do the supplied sample_count
        resp.content_type = falcon.MEDIA_JSON
        resp.status = falcon.HTTP_400

        status = None
        if q.full():
            status = 'Request queue is full'
        else:
            status = sdr_recorder.validate_request(
                arguments.freq_excluded, center_freq, sample_count, sample_rate)

        if status is None:
            q.put({
                'center_freq': int(center_freq),
                'sample_count': int(sample_count),
                'sample_rate': int(sample_rate)})
            status = 'Requsted recording'
            resp.status = falcon.HTTP_200

        resp.text = json.dumps({'status': status})


class API:

    def __init__(self, start_app=True):
        self.mqtt_reporter = MQTTReporter(
            arguments.name, arguments.mqtt_server, ORCHESTRATOR, True)
        self.main(start_app)

    def run_recorder(self, record_func, q):
        logging.info('run recorder')
        while True:
            if arguments.enable_rssi:
                self.serve_rssi(arguments, q)
            else:
                self.serve_recording(arguments, record_func, q)
            time.sleep(5)

    @staticmethod
    def record(center_freq, sample_count, sample_rate=20e6):
        return sdr_recorder.run_recording(
            arguments.path, sample_rate, sample_count, center_freq, arguments.gain, arguments.agc, arguments.rxb, arguments.sigmf, arguments.sdr, arguments.antenna)

    def serve_recording(self, arguments, record_func, q):
        logging.info('serving recordings')
        start_time = time.time()
        while True:
            logging.info('awaiting request')
            record_args = q.get()
            logging.info(f'got a request: {record_args}')
            record_status = record_func(**record_args)
            if record_status == -1:
                # TODO this only kills the thread, not the main process
                break
            record_args.update(vars(arguments))
            self.mqtt_reporter.publish('gamutrf/record', record_args)
            self.mqtt_reporter.log(arguments.path, 'record', start_time, record_args)

    def report_rssi(self, args, record_args, reported_rssi, reported_time, start_time):
        logging.info(
            f'reporting RSSI {reported_rssi} for {record_args["center_freq"]}')
        record_args.update({
            'rssi': reported_rssi,
            'time': reported_time})
        record_args.update(vars(args))
        self.mqtt_reporter.publish('gamutrf/rssi', record_args)
        self.mqtt_reporter.log(args.path, 'rssi', start_time, record_args)

    def process_rssi(self, args, record_args, sock, q):
        last_rssi_time = 0
        start_time = time.time()
        while q.empty():
            rssi_raw, _ = sock.recvfrom(FLOAT_SIZE)
            rssi = struct.unpack('f', rssi_raw)[0]
            if rssi < args.rssi_threshold:
                continue
            if rssi > MAX_RSSI:
                continue
            now = time.time()
            now_diff = now - last_rssi_time
            if now_diff < args.rssi_interval:
                continue
            self.report_rssi(args, record_args, rssi, now, start_time)
            last_rssi_time = now

    def serve_rssi(self, args, q):
        record_args = q.get()
        logging.info(f'got request {record_args}')
        center_freq = int(record_args['center_freq'])
        rssi_server = BirdsEyeRSSI(
            arguments, record_args['sample_rate'], center_freq,
            agc=arguments.agc, rssi_throttle=arguments.rssi_throttle)
        rssi_server.start()
        logging.info(
            f'serving RSSI for {center_freq}Hz over threshold {args.rssi_threshold} with AGC {args.agc}')
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((RSSI_UDP_ADDR, RSSI_UDP_PORT))
            self.process_rssi(args, record_args, sock, q)
        logging.info('RSSI stream stopped')
        rssi_server.stop()
        rssi_server.wait()

    @staticmethod
    def paths():
        return ['', '/info', '/record/{center_freq}/{sample_count}/{sample_rate}']

    @staticmethod
    def version():
        return '/v1'

    def routes(self):
        p = self.paths()
        endpoints = Endpoints()
        info = Info()
        record = Record()
        funcs = [endpoints, info, record]
        return dict(zip(p, funcs))

    def create_app(self):
        cors = CORS(allow_all_origins=True)
        self.app = falcon.App(middleware=[cors.middleware])
        logging.info('adding API routes')
        r = self.routes()
        for route in r:
            self.app.add_route(self.version()+route, r[route])

    def main(self, start_app):
        if start_app:
            logging.info('starting recorder thread')
            recorder_thread = threading.Thread(
                target=self.run_recorder, args=(self.record, q))
            recorder_thread.start()

            self.create_app()

            logging.info('starting API thread')
            bjoern.run(self.app, '0.0.0.0', arguments.port)

            recorder_thread.join()
