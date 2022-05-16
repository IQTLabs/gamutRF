import argparse
import datetime
import json
import logging
import os
import queue
import socket
import struct
import subprocess
import threading
import time

import bjoern
import falcon
from falcon_cors import CORS
from gnuradio import analog, blocks, gr, network, soapy, uhd
import gpsd
import paho.mqtt.client as mqtt
import sigmf

from gamutrf.__init__ import __version__
from gamutrf.utils import ETTUS_ARGS, ETTUS_ANT
from gamutrf.sigwindows import parse_freq_excluded, freq_excluded


class BirdsEyeRSSI(gr.top_block):

    def __init__(self, args, samp_rate, center_freq):
        gr.top_block.__init__(self, 'BirdsEyeRSSI', catch_exceptions=True)

        self.threshold = args.rssi_threshold
        self.samp_rate = samp_rate
        self.gain = args.gain
        self.center_freq = center_freq

        dev = f'driver={args.sdr}'
        stream_args = ''
        tune_args = ['']
        settings = ['']

        # TODO: use common code with grscan.py
        if args.sdr == 'ettus':
            self.source_0 = uhd.usrp_source(
                    ','.join(('', '')),
                    uhd.stream_args(
                        cpu_format='fc32',
                        args=ETTUS_ARGS,
                        channels=list(range(0, 1)),
                    ),
            )
            self.source_0.set_time_now(
                uhd.time_spec(time.time()), uhd.ALL_MBOARDS)
            self.source_0.set_antenna(ETTUS_ANT, 0)
            self.source_0.set_samp_rate(self.samp_rate)
        else:
            self.source_0 = soapy.source(dev, 'fc32', 1, '', stream_args, tune_args, settings)
            self.source_0.set_sample_rate(0, self.samp_rate)
            self.source_0.set_bandwidth(0, 0.0)
            self.source_0.set_frequency(0, self.center_freq)
            self.source_0.set_frequency_correction(0, 0)

        self.source_0.set_gain(0, min(max(self.gain, -1.0), 60.0))

        self.network_udp_sink_0 = network.udp_sink(gr.sizeof_float, 1, RSSI_UDP_ADDR, RSSI_UDP_PORT, 0, 1472, False)
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(1, 1, 0)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_ff(10)
        self.blocks_moving_average_xx_0 = blocks.moving_average_ff(256, 1, 4000, 1)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)
        self.blocks_add_const_vxx_0 = blocks.add_const_ff(-60)
        self.analog_pwr_squelch_xx_0 = analog.pwr_squelch_cc(self.threshold, 5e-4, 1000, True)

        self.connect((self.analog_pwr_squelch_xx_0, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self.blocks_add_const_vxx_0, 0), (self.network_udp_sink_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_moving_average_xx_0, 0))
        self.connect((self.blocks_moving_average_xx_0, 0), (self.blocks_nlog10_ff_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.blocks_add_const_vxx_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.source_0, 0), (self.analog_pwr_squelch_xx_0, 0))


class SDRRecorder:

    def record_args(self, sample_file, sample_rate, sample_count, center_freq, gain, agc, rxb):
        raise NotImplementedError


class EttusRecorder(SDRRecorder):

    def record_args(self, sample_file, sample_rate, sample_count, center_freq, gain, _agc, rxb):
        # Ettus "nsamps" API has an internal limit, so translate "stream for druation".
        duration = round(sample_count / sample_rate)
        return [
            '/usr/local/bin/mt_rx_samples_to_file',
            '--file', sample_file + '.zst',
            '--rate', str(sample_rate),
            '--bw', str(sample_rate),
            '--duration', str(duration),
            '--freq', str(center_freq),
            '--gain', str(gain),
            '--args', ETTUS_ARGS,
            '--ant', ETTUS_ANT,
            '--spb', str(rxb)]


class BladeRecorder(SDRRecorder):

    def record_args(self, sample_file, sample_rate, sample_count, center_freq, gain, agc, _rxb):
        gain_args = [
           '-e', 'set agc rx off',
           '-e', f'set gain rx {gain}',
        ]
        if agc:
            gain_args = [
                '-e', 'set agc rx on',
            ]
        return ['bladeRF-cli' ] + gain_args + [
            '-e', f'set samplerate rx {sample_rate}',
            '-e', f'set bandwidth rx {sample_rate}',
            '-e', f'set frequency rx {center_freq}',
            '-e', f'rx config file={sample_file} format=bin n={sample_count}',
            '-e', 'rx start',
            '-e', 'rx wait']


class LimeRecorder(SDRRecorder):

    def record_args(self, sample_file, sample_rate, sample_count, center_freq, gain, agc, _rxb):
        gain_args = []
        if gain:
            gain_args = [
                '-g', f'{gain}',
            ]
        return ['/usr/local/bin/LimeStream'] + gain_args + [
            '-f', f'{center_freq}',
            '-s', f'{sample_rate}',
            '-C', f'{sample_count}',
            '-r', f'{sample_file}',
        ]

RECORDER_MAP = {
    'ettus': EttusRecorder,
    'bladerf': BladeRecorder,
    'lime': LimeRecorder,
}

FLOAT_SIZE = 4
RSSI_UDP_ADDR = '127.0.0.1'
RSSI_UDP_PORT = 2001
MIN_RSSI = -100
MAX_RSSI = 100
MIN_SAMPLE_RATE = int(1e6)
MAX_SAMPLE_RATE = int(30 * 1e6)

WORKER_NAME = os.getenv('WORKER_NAME', socket.gethostbyname(socket.gethostname()))
ORCHESTRATOR = os.getenv('ORCHESTRATOR', 'orchestrator')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--loglevel', '-l', help='Set logging level',
    choices=['critical', 'error', 'warning', 'info', 'debug'], default='info')
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
    '--sdr', '-s', help='Specify SDR to record with (ettus, lime or bladerf)',
    choices=list(RECORDER_MAP.keys()), default='ettus')
parser.add_argument(
    '--freq_excluded', '-e', help='Freq range to exclude in MHz (e.g. "100-200")',
    action='append', default=[])
parser.add_argument(
    '--gain', '-g', help='Gain in dB',
    default=30, type=int)
parser.add_argument(
    '--rxb', help='Receive buffer size',
    default=int(1024 * 1024 * 10), type=int)
parser.add_argument(
    '--qsize', help='Max request queue size',
    default=int(2), type=int)
parser.add_argument('--mqtt_server', default=ORCHESTRATOR, type=str,
                     help='MQTT server to report RSSI')
parser.add_argument('--rssi_interval', default=1.0, type=float,
                     help='rate limit in seconds for RSSI updates')
parser.add_argument('--rssi_threshold', default=-45, type=float,
                     help='RSSI reporting threshold')
arg_parser = parser.add_mutually_exclusive_group(required=False)
arg_parser.add_argument('--agc', dest='agc', action='store_true', default=True, help='use AGC')
arg_parser.add_argument('--no-agc', dest='agc', action='store_false', help='do not use AGC')
sigmf_parser = parser.add_mutually_exclusive_group(required=False)
sigmf_parser.add_argument('--sigmf', dest='sigmf', action='store_true', help='add sigmf meta file')
sigmf_parser.add_argument('--no-sigmf', dest='sigmf', action='store_false', help='do not add sigmf meta file')
rssi_parser = parser.add_mutually_exclusive_group(required=False)
rssi_parser.add_argument('--rssi', dest='rssi', action='store_true', help='get RSSI values')
rssi_parser.add_argument('--no-rssi', dest='rssi', action='store_false', help='do not get RSSI values')
parser.set_defaults(feature=True)

arguments = parser.parse_args()
q = queue.Queue(arguments.qsize)

sdr_recorder = RECORDER_MAP[arguments.sdr]()

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

        for arg in [center_freq, sample_count, sample_rate]:
            try:
                int(float(arg))
            except ValueError:
                resp.text = json.dumps({'status': 'Invalid values in request'})
                return
        if freq_excluded(center_freq, parse_freq_excluded(arguments.freq_excluded)):
            resp.text = json.dumps({'status': 'Requested frequency is excluded'})
            return
        if int(sample_rate) < MIN_SAMPLE_RATE or int(sample_rate) > MAX_SAMPLE_RATE:
            resp.text = json.dumps({'status': f'sample rate {sample_rate} out of range {MIN_SAMPLE_RATE} to {MAX_SAMPLE_RATE}'})
            return
        duration_sec = int(sample_count) / int(sample_rate)
        if duration_sec < 1:
           resp.text = json.dumps({'status': 'cannot record for less than 1 second'})
           return
        if q.full():
            resp.text = json.dumps({'status': 'Request queue is full'})
            return
        q.put({
            'center_freq': int(center_freq),
            'sample_count': int(sample_count),
            'sample_rate': int(sample_rate)})
        resp.text = json.dumps({'status': 'Requested recording'}, indent=2)
        resp.status = falcon.HTTP_200


class API:

    def __init__(self):
        cors = CORS(allow_all_origins=True)
        self.app = falcon.App(middleware=[cors.middleware])
        self.mqttc = None
        self.main()

    def run_recorder(self, record_func, q):
        logging.info('run recorder')
        while True:
            if arguments.rssi:
                # TODO: this is only get called the first time then be stuck in a loop, ignoring the rest of the queue
                # TODO: may have to run in subprocess to avoid overruns.
                record_args = q.get()
                logging.info(f'got request {record_args}')
                rssi_server = BirdsEyeRSSI(arguments, record_args['sample_rate'], record_args['center_freq'])
                rssi_server.start()
                self.serve_rssi(arguments, record_args)
            else:
                self.serve_recording(arguments, record_func, q)
            time.sleep(5)

    @staticmethod
    def record(center_freq, sample_count, sample_rate=20e6):
        agc = arguments.agc
        gain = arguments.gain
        rxb = arguments.rxb
        epoch_time = str(int(time.time()))
        meta_time = datetime.datetime.utcnow().isoformat() + 'Z'
        sample_type = 's16'
        sample_file = os.path.join(
            arguments.path, f'gamutrf_recording{epoch_time}_{int(center_freq)}Hz_{int(sample_rate)}sps.{sample_type}')
        args = sdr_recorder.record_args(sample_file, sample_rate, sample_count, center_freq, gain, agc, rxb)
        logging.info('starting recording: %s', args)
        record_status = -1
        try:
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
        except subprocess.CalledProcessError as err:
            logging.debug('record failed: %s', err)
        logging.info('record status: %d', record_status)
        return record_status

    def connect_mqtt(self, args):
        if args.mqtt_server:
            logging.info(f'connecting to {args.mqtt_server}')
            self.mqttc = mqtt.Client()
            self.mqttc.connect(args.mqtt_server)
            self.mqttc.loop_start()
            gpsd.connect(host=ORCHESTRATOR, port=2947)

    def make_record_packet(self, record_args):
        packet = gpsd.get_current()
        record_args["position"] = packet.position()
        record_args["altitude"] = packet.altitude()
        record_args["gps_time"] = packet.get_time().timestamp()
        record_args["map_url"] = packet.map_url()
        return json.dumps(record_args)

    def serve_recording(self, arguments, record_func, q):
        logging.info('serving recordings')
        try:
            self.connect_mqtt(arguments)
            while True:
                logging.info('awaiting request')
                record_args = q.get()
                logging.info(f'got a request: {record_args}')
                record_status = record_func(**record_args)
                if record_status == -1:
                    # TODO this only kills the thread, not the main process
                    break
                if self.mqttc:
                    record_args = self.make_record_packet(record_args)
                    self.mqttc.publish("gamutrf/record", record_args)
        except (ConnectionRefusedError, mqtt.WebsocketConnectionError, ValueError) as e:
            logging.error(f'failed to report to MQTT {arguments.mqtt_server}: {e}')

    def report_rssi(self, args, record_args, reported_rssi):
        logging.info(f'reporting RSSI {reported_rssi}')
        if not self.mqttc:
            return
        try:
            self.connect_mqtt(args)
            record_args = self.make_record_packet({'rssi': reported_rssi})
            self.mqttc.publish('gamutrf/record', record_args)
            self.mqttc.loop_stop()
        except (ConnectionRefusedError, mqtt.WebsocketConnectionError, ValueError) as err:
            logging.error(f'failed to report RSSI to MQTT {arguments.mqtt_server}: {err}')

    def process_rssi(self, args, record_args, sock):
        last_rssi_time = 0
        self.connect_mqtt(args)
        while True:
            rssi_raw, _ = sock.recvfrom(FLOAT_SIZE)
            rssi = struct.unpack('f', rssi_raw)[0]
            if rssi < MIN_RSSI or rssi > MAX_RSSI:
                continue
            now = time.time()
            now_diff = now - last_rssi_time
            if now_diff < args.rssi_interval:
                continue
            self.report_rssi(args, record_args, rssi)
            last_rssi_time = now

    def serve_rssi(self, args, record_args):
        logging.info('serving RSSI')
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((RSSI_UDP_ADDR, RSSI_UDP_PORT))
            self.process_rssi(args, record_args, sock)

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
        logging.info('starting recorder thread')
        recorder_thread = threading.Thread(
            target=self.run_recorder, args=(self.record, q))
        recorder_thread.start()

        logging.info('adding API routes')
        r = self.routes()
        for route in r:
            self.app.add_route(self.version()+route, r[route])

        logging.info('starting API thread')
        bjoern.run(self.app, '0.0.0.0', arguments.port)
        recorder_thread.join()
