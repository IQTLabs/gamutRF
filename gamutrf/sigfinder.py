#!/usr/bin/python3

import argparse
import json
import logging
import os
import socket
import threading
import time

import bjoern
import falcon
import jinja2
import pandas as pd
from prometheus_client import start_http_server, Counter, Gauge
import requests
import schedule

from gamutrf.sigwindows import calc_db
from gamutrf.sigwindows import choose_record_signal
from gamutrf.sigwindows import choose_recorders
from gamutrf.sigwindows import find_sig_windows
from gamutrf.sigwindows import parse_freq_excluded
from gamutrf.sigwindows import get_center

SOCKET_TIMEOUT = 1.0
ROLLOVERHZ = 100e6
PEAK_DBS = {}


def falcon_response(resp, text, status):
    resp.status = status
    resp.text = text
    resp.content_type = 'text/html'


def ok_response(resp, text='ok!'):
    falcon_response(resp, text=text, status=falcon.HTTP_200)


def error_response(resp, text='error!'):
    falcon_response(resp, text=text, status=falcon.HTTP_500)


def load_template(name):
    path = os.path.join('templates', name)
    with open(os.path.abspath(path), 'r') as fp:
        return jinja2.Template(fp.read())


class ActiveRequests:
    def on_get(self, req, resp):
        all_jobs = schedule.get_jobs()
        ok_response(resp, f'{all_jobs}')


class ScannerForm:
    def on_get(self, req, resp):
        template = load_template('scanner_form.html')
        ok_response(resp, template.render(bins=PEAK_DBS))


class Result:
    def on_post(self, req, resp):
        # TODO validate input
        try:
            recorder = f'http://{req.media["worker"]}:8000/'
            signal_hz = int(int(req.media['frequency']) * 1e6)
            record_bps = int(int(req.media['bandwidth']) * (1024 * 1024))
            record_samples = int(record_bps * int(req.media['duration']))
            recorder_args = f'record/{signal_hz}/{record_samples}/{record_bps}'
            timeout = int(req.media['duration'])
            response = None
            if int(req.media['repeat']) == -1:
                schedule.every(timeout).seconds.do(run_threaded, record, recorder=recorder, recorder_args=recorder_args, timeout=timeout).tag(f'{recorder}{recorder_args}-{timeout}')
                ok_response(resp)
            else:
                response = recorder_req(recorder, recorder_args, timeout)
                time.sleep(timeout)
                for _ in range(int(req.media['repeat'])):
                    response = recorder_req(recorder, recorder_args, timeout)
                    time.sleep(timeout)
                if response:
                    ok_response(resp)
                else:
                    ok_response(resp, f'Request {recorder} {recorder_args} failed.')
        except Exception as e:
            error_response(resp, f'{e}')


def record(recorder, recorder_args, timeout):
    recorder_req(recorder, recorder_args, timeout)


def run_threaded(job_func, recorder, recorder_args, timeout):
    job_thread = threading.Thread(target=job_func, args=(recorder, recorder_args, timeout,))
    job_thread.start()


def init_prom_vars():
    prom_vars = {
        'last_bin_freq_time': Gauge('last_bin_freq_time', 'epoch time last signal in each bin', labelnames=('bin_mhz',)),
        'worker_record_request': Gauge('worker_record_request', 'record requests made to workers', labelnames=('worker',)),
        'freq_power': Gauge('freq_power', 'bin frequencies and db over time', labelnames=('bin_freq',)),
        'new_bins': Counter('new_bins', 'frequencies of new bins', labelnames=('bin_freq',)),
        'old_bins': Counter('old_bins', 'frequencies of old bins', labelnames=('bin_freq',)),
        'bin_freq_count': Counter('bin_freq_count', 'count of signals in each bin', labelnames=('bin_mhz',)),
        'frame_counter': Counter('frame_counter', 'number of frames processed'),
    }
    return prom_vars


def update_prom_vars(peak_dbs, new_bins, old_bins, prom_vars):
    freq_power = prom_vars['freq_power']
    new_bins_prom = prom_vars['new_bins']
    old_bins_prom = prom_vars['old_bins']
    for freq in peak_dbs:
        freq_power.labels(bin_freq=freq).set(peak_dbs[freq])
    for nbin in new_bins:
        new_bins_prom.labels(bin_freq=nbin).inc()
    for obin in old_bins:
        old_bins_prom.labels(bin_freq=obin).inc()


def process_fft(args, prom_vars, ts, fftbuffer, lastbins):
    global PEAK_DBS
    tsc = time.ctime(ts)
    logging.info(f'new frame at {tsc}')
    df = pd.DataFrame(fftbuffer, columns=['ts', 'freq', 'db'])
    df['freq'] /= 1e6
    df = calc_db(df)
    monitor_bins = set()
    peak_dbs = {}
    bin_freq_count = prom_vars['bin_freq_count']
    last_bin_freq_time = prom_vars['last_bin_freq_time']
    freq_start_mhz = args.freq_start / 1e6
    freq_end_mhz = args.freq_end / 1e6
    for signal in find_sig_windows(df, window=args.window, threshold=args.threshold):
        start_freq, end_freq = signal[:2]
        peak_db = signal[-1]
        center_freq = start_freq + ((end_freq - start_freq) / 2)
        if center_freq < freq_start_mhz or center_freq > freq_end_mhz:
            print(f'ignoring {center_freq}')
            continue
        center_freq = get_center(center_freq, freq_start_mhz, args.bin_mhz, args.record_bw_mbps)
        bin_freq_count.labels(bin_mhz=center_freq).inc()
        last_bin_freq_time.labels(bin_mhz=ts).set(ts)
        monitor_bins.add(center_freq)
        peak_dbs[center_freq] = peak_db
    logging.info('current bins %f to %f MHz: %s',
                 df['freq'].min(), df['freq'].max(), sorted(peak_dbs.items()))
    PEAK_DBS = sorted(peak_dbs.items())
    new_bins = monitor_bins - lastbins
    if new_bins:
        logging.info('new bins: %s', sorted(new_bins))
    old_bins = lastbins - monitor_bins
    if old_bins:
        logging.info('old bins: %s', sorted(old_bins))
    update_prom_vars(peak_dbs, new_bins, old_bins, prom_vars)
    return monitor_bins


def recorder_req(recorder, recorder_args, timeout):
    url = f'{recorder}/v1/{recorder_args}'
    try:
        req = requests.get(url, timeout=timeout)
        logging.debug(str(req))
        return req
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as err:
        logging.debug(str(err))
        return None


def get_freq_exclusions(args):
    recorder_freq_exclusions = {}
    for recorder in args.recorder:
        req = recorder_req(recorder, 'info', args.record_secs)
        if req is None or req.status_code != 200:
            continue
        excluded = json.loads(req.text).get('freq_excluded', None)
        if excluded is None:
            continue
        recorder_freq_exclusions[recorder] = parse_freq_excluded(excluded)
    return recorder_freq_exclusions


def call_record_signals(args, lastbins_history, prom_vars):
    if lastbins_history:
        signals = []
        for bins in lastbins_history:
            signals.extend(list(bins))
        recorder_freq_exclusions = get_freq_exclusions(
            args)
        recorder_count = len(recorder_freq_exclusions)
        record_signals = choose_record_signal(
            signals, recorder_count)
        for signal, recorder in choose_recorders(record_signals, recorder_freq_exclusions):
            signal_hz = int(signal * 1e6)
            record_bps = int(args.record_bw_mbps * (1024 * 1024))
            record_samples = int(
                record_bps * args.record_secs)
            recorder_args = f'record/{signal_hz}/{record_samples}/{record_bps}'
            resp = recorder_req(
                recorder, recorder_args, args.record_secs)
            if resp:
                worker_record_request = prom_vars['worker_record_request']
                worker_record_request.labels(worker=recorder).set(signal_hz)


def process_fft_lines(args, prom_vars, sock, ext):
    lastfreq = 0
    fftbuffer = []
    lastbins_history = []
    lastbins = set()
    frame_counter = prom_vars['frame_counter']
    txt_buf = ''

    while True:
        if os.path.exists(args.log):
            logging.info(f'{args.log} exists, will append first')
            mode = 'ab'
        else:
            logging.info(f'opening {args.log}')
            mode = 'wb'
        openlogts = int(time.time())
        with open(args.log, mode=mode) as l:
            while True:
                schedule.run_pending()
                try:
                    sock.settimeout(SOCKET_TIMEOUT)
                    sock_txt, _ = sock.recvfrom(2048)
                except socket.timeout:
                    logging.info('timeout receiving FFT from scanner - retrying')
                    continue
                if not len(sock_txt):
                    return
                txt_buf += sock_txt.decode('utf8')
                lines = txt_buf.splitlines()
                if txt_buf.endswith('\n'):
                    txt_buf = ''
                else:
                    txt_buf = lines[-1]
                    lines = lines[:-1]
                rotatelognow = False
                now = int(time.time())
                for line in lines:
                    try:
                        ts, freq, pw = [float(x) for x in line.strip().split()]
                    except ValueError:
                        continue
                    if pw < 0 or pw > 1:
                        continue
                    if freq < 0 or freq > 10e9:
                        continue
                    if abs(now - ts) > 60:
                        continue
                    l.write(line.encode('utf8'))
                    rollover = abs(freq - lastfreq) > ROLLOVERHZ and fftbuffer
                    fftbuffer.append((ts, freq, pw))
                    lastfreq = freq
                    if rollover:
                        frame_counter.inc()
                        lastbins = process_fft(args, prom_vars, ts, fftbuffer, lastbins)
                        if lastbins:
                            lastbins_history = [lastbins] + lastbins_history
                            lastbins_history = lastbins_history[:args.history]
                        fftbuffer = []
                        call_record_signals(args, lastbins_history, prom_vars)
                        if now - openlogts > args.rotatesecs:
                            rotatelognow = True
                if rotatelognow:
                    break
        new_log = args.log.replace(ext, f'{openlogts}{ext}')
        os.rename(args.log, new_log)


def find_signals(args, prom_vars):
    try:
        ext = args.log[args.log.rindex('.'):]
    except ValueError:
        logging.fatal(f'cannot parse extension from {args.log}')

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setblocking(False)
        sock.bind((args.logaddr, args.logport))
        process_fft_lines(args, prom_vars, sock, ext)


def main():
    parser = argparse.ArgumentParser(
        description='watch a scan UDP stream and find signals')
    parser.add_argument('--log', default='scan.log', type=str,
                        help='base path for scan logging')
    parser.add_argument('--rotatesecs', default=3600, type=int,
                        help='rotate scan log after this many seconds')
    parser.add_argument('--logaddr', default='127.0.0.1', type=str,
                        help='UDP stream address')
    parser.add_argument('--logport', default=8001, type=int,
                        help='UDP stream port')
    parser.add_argument('--bin_mhz', default=20, type=int,
                        help='monitoring bin width in MHz')
    parser.add_argument('--window', default=4, type=int,
                        help='signal finding sample window size')
    parser.add_argument('--threshold', default=1.5, type=float,
                        help='signal finding threshold')
    parser.add_argument('--history', default=50, type=int,
                        help='number of frames of signal history to keep')
    parser.add_argument('--recorder', action='append', default=[],
                        help='SDR recorder base URLs (e.g. http://host:port/, multiples can be specified)')
    parser.add_argument('--record_bw_mbps', default=20, type=int,
                        help='record bandwidth in mbps')
    parser.add_argument('--record_secs', default=10, type=int,
                        help='record time duration in seconds')
    parser.add_argument('--promport', dest='promport', type=int, default=9000,
                        help='Prometheus client port')
    parser.add_argument(
        '--freq-end', dest='freq_end', type=float, default=float(1e9),
        help='Set freq_end [default=%(default)r]')
    parser.add_argument(
        '--freq-start', dest='freq_start', type=float, default=float(100e6),
        help='Set freq_start [default=%(default)r]')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    prom_vars = init_prom_vars()
    start_http_server(args.promport)
    x = threading.Thread(target=find_signals, args=(args, prom_vars,))
    x.start()
    app = falcon.App()
    scanner_form = ScannerForm()
    result = Result()
    active_requests = ActiveRequests()
    app.add_route('/', scanner_form)
    app.add_route('/result', result)
    app.add_route('/requests', active_requests)
    bjoern.run(app, '0.0.0.0', 80)


if __name__ == '__main__':
    main()
