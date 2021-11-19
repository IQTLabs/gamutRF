import argparse
import datetime
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
import sigmf

from gamutrf.__init__ import __version__
from gamutrf.sigwindows import parse_freq_excluded, freq_excluded


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
    '--sdr', '-s', help='Specify SDR to record with (ettus, lime or bladerf)',
    choices=['bladerf', 'ettus', 'lime'], default='ettus')
parser.add_argument(
    '--freq_excluded', '-e', help='Freq range to exclude in MHz (e.g. "100-200")',
    action='append', default=[])
sigmf_parser = parser.add_mutually_exclusive_group(required=False)
sigmf_parser.add_argument('--sigmf', dest='sigmf', action='store_true', help='add sigmf meta file')
sigmf_parser.add_argument('--no-sigmf', dest='sigmf', action='store_false', help='do not add sigmf meta file')
parser.set_defaults(feature=True)

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

        for arg in [center_freq, sample_count, sample_rate]:
            try:
                int(float(arg))
            except ValueError:
                resp.text = json.dumps({'status': 'Invalid values in request'})
                return
        if freq_excluded(center_freq, parse_freq_excluded(arguments.freq_excluded)):
            resp.text = json.dumps({'status': 'Requested frequency is excluded'})
            return
        q.put({'center_freq': center_freq,
              'sample_count': sample_count, 'sample_rate': sample_rate})
        resp.text = json.dumps({'status': 'Requested recording'}, indent=2)
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
    @staticmethod
    def raw2grraw(in_file, gr_file, sample_rate, in_file_bits=16, in_file_fmt='signed-integer'):
        raw_args = ['-t', 'raw', '-r', str(sample_rate), '-c', str(1)]
        return subprocess.check_call(
            ['sox'] + raw_args + ['-b', str(in_file_bits), '-e', in_file_fmt, in_file] + raw_args + ['-e', 'float', gr_file])

    @staticmethod
    def record(center_freq, sample_count, sample_rate=20e6, gain=0, agc=True):
        epoch_time = str(int(time.time()))
        meta_time = datetime.datetime.utcnow().isoformat() + 'Z'
        sample_type = 's16'
        sample_file = os.path.join(
            arguments.path, f'gamutrf_recording{epoch_time}_{int(center_freq)}Hz_{int(sample_rate)}sps.{sample_type}')
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
        elif arguments.sdr == 'lime':
            gain_args = []
            if gain:
                gain_args = [
                    '-g', f'{gain}',
                ]
            args = [
                '/usr/local/bin/LimeStream',
            ] + gain_args + [
                '-f', f'{center_freq}',
                '-s', f'{sample_rate}',
                '-C', f'{sample_count}',
                '-r', f'{sample_file}',
            ]
        else:
            logging.error('Invalid SDR, not recording')
            return -1
        record_status = subprocess.check_call(args)
        if arguments.sigmf:
            meta = sigmf.SigMFFile(
                data_file = sample_file,
                global_info = {
                    sigmf.SigMFFile.DATATYPE_KEY: sample_type,
                    sigmf.SigMFFile.SAMPLE_RATE_KEY: sample_rate,
                    sigmf.SigMFFile.VERSION_KEY: sigmf.__version__,
                })
            meta.add_capture(0, metadata={
                sigmf.SigMFFile.FREQUENCY_KEY: center_freq,
                sigmf.SigMFFile.DATETIME_KEY: meta_time,
            })
            meta.tofile(sample_file + '.sigmf-meta')
        return record_status

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
