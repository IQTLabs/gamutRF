import argparse
import json
import logging
import os
import queue
import subprocess
import threading
import time

import bjoern
import falcon
from falcon_cors import CORS

from gamutrf.__init__ import __version__


q = queue.Queue()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--loglevel', '-l', help='Set logging level',
    choices=['critical', 'error', 'warning', 'info', 'debug'], default='info')
parser.add_argument(
    '--path', '-P', help='Path prefix for writing out samples to',
    type=str, default='/data/gamutrf')
parser.add_argument(
    '--port', '-p', help='Port to run the API webserver on', type=int,
    default=8000)
parser.add_argument(
    '--sdr', '-s', help='Specify SDR to record with (ettus or bladerf)',
    choices=['bladerf', 'ettus'], default='ettus')
arguments = parser.parse_args()

level_int = {'CRITICAL': 50, 'ERROR': 40, 'WARNING': 30, 'INFO': 20,
             'DEBUG': 10}
level = level_int.get(arguments.loglevel.upper(), 0)
logging.basicConfig(level=level)


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
            {'version': __version__, 'sdr': arguments.sdr, 'path_prefix': arguments.path})
        resp.content_type = falcon.MEDIA_TEXT
        resp.status = falcon.HTTP_200


class Record:

    @staticmethod
    def on_get(_req, resp, center_freq, sample_count, sample_rate):
        # TODO check if chosen SDR can do the supplied center_freq/sample_count
        for arg in [center_freq, sample_count, sample_rate]:
            try:
                int(float(arg))
            except ValueError:
                resp.text = json.dumps({'status': 'Invalid values in request'})
                resp.content_type = falcon.MEDIA_JSON
                resp.status = falcon.HTTP_400
                return
        q.put({'center_freq': center_freq,
              'sample_count': sample_count, 'sample_rate': sample_rate})
        resp.text = json.dumps({'status': 'Requested recording'}, indent=2)
        resp.content_type = falcon.MEDIA_JSON
        resp.status = falcon.HTTP_200


class API:

    def __init__(self):
        cors = CORS(allow_all_origins=True)
        self.app = falcon.App(middleware=[cors.middleware])
        self.main()

    @staticmethod
    def run_recorder(record_func, q):
        while True:
            record_args = q.get()
            record_func(**record_args)

    # Convert I/Q sample recording to "gnuradio" I/Q format (float)
    # Default input format is signed, 16 bit I/Q (bladeRF-cli default)
    def raw2grraw(in_file, gr_file, sample_rate, in_file_bits=16, in_file_fmt='signed-integer'):
        raw_args = ['-t', 'raw', '-r', str(sample_rate), '-c', str(1)]
        return subprocess.check_call(
            ['sox'] + raw_args + ['-b', str(in_file_bits), '-e', in_file_fmt, in_file] + raw_args + ['-e', 'float', gr_file])

    @staticmethod
    def record(center_freq, sample_count, sample_rate=20e6, gain=0, agc=True):
        epoch_time = str(int(time.time()))
        sample_file = os.path.join(
            arguments.path, f'gamutrf_recording{epoch_time}_{int(center_freq)}Hz_{int(sample_rate)}sps.raw')
        if arguments.sdr == 'ettus':
            args = [
                '/usr/lib/uhd/examples/rx_samples_to_file',
                '--file', sample_file,
                '--rate', str(sample_rate),
                '--bw', str(sample_rate),
                '--nsamps', str(int(sample_count)),
                '--freq', str(center_freq),
                '--gain', str(gain)]
        elif arguments.sdr == 'bladerf':
            gain_args = [
                '-e', 'set agc rx off',
                '-e', f'set gain rx {gain}',
            ]
            if agc:
                gain_args = [
                    '-e', 'set agc rx on',
                ]
            args = [
                'bladeRF-cli',
            ] + gain_args + [
                '-e', f'set samplerate rx {sample_rate}',
                '-e', f'set bandwidth rx {sample_rate}',
                '-e', f'set frequency rx {center_freq}',
                '-e', f'rx config file={sample_file} format=bin n={sample_count}',
                '-e', 'rx start',
                '-e', 'rx wait']
        else:
            logging.error('Invalid SDR, not recording')
            return -1
        return subprocess.check_call(args)

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

    def main(self):

        recorder_thread = threading.Thread(
            target=self.run_recorder, args=(self.record, q))
        recorder_thread.start()

        r = self.routes()
        for route in r:
            self.app.add_route(self.version()+route, r[route])

        bjoern.run(self.app, '0.0.0.0', arguments.port)
        recorder_thread.join()
