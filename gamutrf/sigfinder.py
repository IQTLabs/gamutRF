import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
import threading
import time

import bjoern
import falcon
import jinja2
import numpy as np
import pandas as pd
import requests
import schedule
import zmq
import zstandard

from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import start_http_server

from gamutrf.sigwindows import calc_db
from gamutrf.sigwindows import choose_record_signal
from gamutrf.sigwindows import choose_recorders
from gamutrf.sigwindows import get_center
from gamutrf.sigwindows import graph_fft_peaks
from gamutrf.sigwindows import parse_freq_excluded
from gamutrf.sigwindows import scipy_find_sig_windows
from gamutrf.utils import rotate_file_n, SCAN_FRES

MB = int(1.024e6)
FFT_BUFFER_TIME = 1
BUFF_FILE = "scanfftbuffer.txt.zst"  # nosec
PEAK_TRIGGER = int(os.environ.get("PEAK_TRIGGER", "0"))
PIN_TRIGGER = int(os.environ.get("PIN_TRIGGER", "17"))
if PEAK_TRIGGER == 1:
    import RPi.GPIO as GPIO

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIN_TRIGGER, GPIO.OUT)

PEAK_DBS = {}


def falcon_response(resp, text, status):
    resp.status = status
    resp.text = text
    resp.content_type = "text/html"


def ok_response(resp, text="ok!"):
    falcon_response(resp, text=text, status=falcon.HTTP_200)


def error_response(resp, text="error!"):
    falcon_response(resp, text=text, status=falcon.HTTP_500)


def load_template(name):
    path = os.path.join("templates", name)
    with open(os.path.abspath(path), "r", encoding="utf-8") as fp:
        return jinja2.Template(fp.read())


class ActiveRequests:
    def on_get(self, req, resp):
        all_jobs = schedule.get_jobs()
        ok_response(resp, f"{all_jobs}")


class ScannerForm:
    def on_get(self, req, resp):
        template = load_template("scanner_form.html")
        ok_response(resp, template.render(bins=PEAK_DBS))


class Result:
    def on_post(self, req, resp):
        # TODO validate input
        try:
            recorder = f'http://{req.media["worker"]}:8000/'
            signal_hz = int(int(req.media["frequency"]) * 1e6)
            record_bps = int(int(req.media["bandwidth"]) * MB)
            record_samples = int(record_bps * int(req.media["duration"]))
            recorder_args = f"record/{signal_hz}/{record_samples}/{record_bps}"
            timeout = int(req.media["duration"])
            response = None
            if int(req.media["repeat"]) == -1:
                schedule.every(timeout).seconds.do(
                    run_threaded,
                    record,
                    recorder=recorder,
                    recorder_args=recorder_args,
                    timeout=timeout,
                ).tag(f"{recorder}{recorder_args}-{timeout}")
                ok_response(resp)
            else:
                response = recorder_req(recorder, recorder_args, timeout)
                time.sleep(timeout)
                for _ in range(int(req.media["repeat"])):
                    response = recorder_req(recorder, recorder_args, timeout)
                    time.sleep(timeout)
                if response:
                    ok_response(resp)
                else:
                    ok_response(resp, f"Request {recorder} {recorder_args} failed.")
        except Exception as e:
            error_response(resp, f"{e}")


def record(recorder, recorder_args, timeout):
    recorder_req(recorder, recorder_args, timeout)


def run_threaded(job_func, recorder, recorder_args, timeout):
    job_thread = threading.Thread(
        target=job_func,
        args=(
            recorder,
            recorder_args,
            timeout,
        ),
    )
    job_thread.start()


def init_prom_vars():
    prom_vars = {
        "last_bin_freq_time": Gauge(
            "last_bin_freq_time",
            "epoch time last signal in each bin",
            labelnames=("bin_mhz",),
        ),
        "worker_record_request": Gauge(
            "worker_record_request",
            "record requests made to workers",
            labelnames=("worker",),
        ),
        "freq_power": Gauge(
            "freq_power", "bin frequencies and db over time", labelnames=("bin_freq",)
        ),
        "new_bins": Counter(
            "new_bins", "frequencies of new bins", labelnames=("bin_freq",)
        ),
        "old_bins": Counter(
            "old_bins", "frequencies of old bins", labelnames=("bin_freq",)
        ),
        "bin_freq_count": Counter(
            "bin_freq_count", "count of signals in each bin", labelnames=("bin_mhz",)
        ),
        "frame_counter": Counter("frame_counter", "number of frames processed"),
    }
    return prom_vars


def update_prom_vars(peak_dbs, new_bins, old_bins, prom_vars):
    freq_power = prom_vars["freq_power"]
    new_bins_prom = prom_vars["new_bins"]
    old_bins_prom = prom_vars["old_bins"]
    for freq in peak_dbs:
        freq_power.labels(bin_freq=freq).set(peak_dbs[freq])
    for nbin in new_bins:
        new_bins_prom.labels(bin_freq=nbin).inc()
    for obin in old_bins:
        old_bins_prom.labels(bin_freq=obin).inc()


def process_fft(args, prom_vars, ts, fftbuffer, lastbins, running_df):
    global PEAK_DBS
    df = pd.DataFrame(fftbuffer, columns=["ts", "freq", "db"])
    # resample to SCAN_FRES
    # ...first frequency
    df["freq"] = (df["freq"] / SCAN_FRES).round() * SCAN_FRES / 1e6
    df = df.set_index("freq")
    # ...then power
    df["db"] = df.groupby(["freq"])["db"].mean()
    df = df.reset_index().drop_duplicates(subset=["freq"])
    df = df.sort_values("freq")
    df = calc_db(df)
    freqdiffs = df.freq - df.freq.shift()
    mindiff = freqdiffs.min()
    maxdiff = freqdiffs.max()
    meandiff = freqdiffs.mean()
    logging.info(
        "new frame with %u samples, frequency sample differences min %f mean %f max %f",
        len(df),
        mindiff,
        meandiff,
        maxdiff,
    )
    if meandiff > mindiff * 2:
        logging.warning(
            "mean frequency diff larger than minimum - increase scanner sample rate"
        )
        logging.warning(df[freqdiffs > mindiff * 2])
    if args.fftlog:
        df.to_csv(args.fftlog, sep="\t", index=False)
    monitor_bins = set()
    peak_dbs = {}
    bin_freq_count = prom_vars["bin_freq_count"]
    last_bin_freq_time = prom_vars["last_bin_freq_time"]
    freq_start_mhz = args.freq_start / 1e6
    signals = scipy_find_sig_windows(
        df, width=args.width, prominence=args.prominence, threshold=args.threshold
    )

    if PEAK_TRIGGER == 1 and signals:
        led_sleep = 0.2
        GPIO.output(PIN_TRIGGER, GPIO.HIGH)
        time.sleep(led_sleep)
        GPIO.output(PIN_TRIGGER, GPIO.LOW)

    if running_df is None:
        running_df = df
    else:
        now = time.time()
        running_df = running_df[running_df.ts >= (now - args.running_fft_secs)]
        running_df = pd.concat(running_df, df)
    mean_running_df = running_df[["freq", "db"]].groupby(["freq"]).mean().reset_index()
    sample_count_df = df[["freq"]].copy()
    sample_count_df["freq"] = np.floor(sample_count_df["freq"])
    # nosemgrep
    sample_count_df["size"] = sample_count_df.groupby("freq").transform("size")
    sample_count_df["size"] = abs(
        sample_count_df["size"].mean() - sample_count_df["size"]
    )
    sample_count_df["size"].iat[0] = 0
    sample_count_df["size"].iat[-1] = 0

    if args.fftgraph:
        rotate_file_n(args.fftgraph, args.nfftgraph)
        graph_fft_peaks(args.fftgraph, df, mean_running_df, sample_count_df, signals)

    for peak_freq, peak_db in signals:
        center_freq = get_center(
            peak_freq, freq_start_mhz, args.bin_mhz, args.record_bw_msps
        )
        logging.info(
            "detected peak at %f MHz %f dB, assigned bin frequency %f MHz",
            peak_freq,
            peak_db,
            center_freq,
        )
        bin_freq_count.labels(bin_mhz=center_freq).inc()
        last_bin_freq_time.labels(bin_mhz=ts).set(ts)
        monitor_bins.add(center_freq)
        peak_dbs[center_freq] = peak_db
    logging.info(
        "current bins %f to %f MHz: %s",
        df["freq"].min(),
        df["freq"].max(),
        sorted(peak_dbs.items()),
    )
    PEAK_DBS = sorted(peak_dbs.items())
    new_bins = monitor_bins - lastbins
    if new_bins:
        logging.info("new bins: %s", sorted(new_bins))
    old_bins = lastbins - monitor_bins
    if old_bins:
        logging.info("old bins: %s", sorted(old_bins))
    update_prom_vars(peak_dbs, new_bins, old_bins, prom_vars)
    return monitor_bins


def recorder_req(recorder, recorder_args, timeout):
    url = f"{recorder}/v1/{recorder_args}"
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
        req = recorder_req(recorder, "info", args.record_secs)
        if req is None or req.status_code != 200:
            continue
        excluded = json.loads(req.text).get("freq_excluded", None)
        if excluded is None:
            continue
        recorder_freq_exclusions[recorder] = parse_freq_excluded(excluded)
    return recorder_freq_exclusions


def call_record_signals(args, lastbins_history, prom_vars):
    if lastbins_history:
        signals = []
        for bins in lastbins_history:
            signals.extend(list(bins))
        recorder_freq_exclusions = get_freq_exclusions(args)
        recorder_count = len(recorder_freq_exclusions)
        record_signals = choose_record_signal(
            signals, recorder_count * args.max_recorder_signals
        )
        for signal, recorder in choose_recorders(
            record_signals, recorder_freq_exclusions, args.max_recorder_signals
        ):
            signal_hz = int(signal * 1e6)
            record_bps = int(args.record_bw_msps * MB)
            record_samples = int(record_bps * args.record_secs)
            recorder_args = f"record/{signal_hz}/{record_samples}/{record_bps}"
            resp = recorder_req(recorder, recorder_args, args.record_secs)
            if resp:
                worker_record_request = prom_vars["worker_record_request"]
                worker_record_request.labels(worker=recorder).set(signal_hz)


def zstd_file(uncompressed_file):
    subprocess.check_call(["/usr/bin/zstd", "--force", "--rm", uncompressed_file])


def process_fft_lines(
    args, prom_vars, buff_file, executor, proxy_result, runonce=False
):
    lastfreq = 0
    fftbuffer = []
    lastbins_history = []
    lastbins = set()
    frame_counter = prom_vars["frame_counter"]
    txt_buf = ""
    last_fft_report = 0
    fft_packets = 0
    max_scan_pos = 0
    context = zstandard.ZstdDecompressor()
    running_df = None

    while True:
        if os.path.exists(args.log):
            logging.info(f"{args.log} exists, will append first")
            mode = "a"
        else:
            logging.info(f"opening {args.log}")
            mode = "w"
        openlogts = int(time.time())
        with open(args.log, mode=mode, encoding="utf-8") as l:
            while True:
                if not proxy_result.running():
                    logging.error(
                        "FFT proxy stopped running: %s", proxy_result.result()
                    )
                    sys.exit(-1)
                now = int(time.time())
                if now - last_fft_report > FFT_BUFFER_TIME * 2:
                    logging.info(
                        "received %u FFT packets, last freq %f MHz",
                        fft_packets,
                        lastfreq / 1e6,
                    )
                    fft_packets = 0
                    last_fft_report = now
                if os.path.exists(buff_file):
                    logging.info(
                        "read %u bytes of FFT data", os.stat(buff_file).st_size
                    )
                    with context.stream_reader(open(buff_file, "rb")) as bf:
                        txt_buf += bf.read().decode("utf8")
                    fft_packets += 1
                    os.remove(buff_file)
                else:
                    schedule.run_pending()
                    sleep_time = 1
                    time.sleep(sleep_time)
                    continue
                lines = txt_buf.splitlines()
                if not len(lines) > 1:
                    continue
                if txt_buf.endswith("\n"):
                    l.write(txt_buf)
                    txt_buf = ""
                elif lines:
                    last_line = lines[-1]
                    l.write(txt_buf[: -len(last_line)])
                    txt_buf = last_line
                    lines = lines[:-1]
                try:
                    df = pd.DataFrame(
                        [line.strip().split() for line in lines],
                        columns=["ts", "freq", "pw"],
                        dtype=float,
                    )
                except ValueError as err:
                    logging.error(str(err))
                    continue
                df = df[(df.pw > 0) & (df.pw <= args.max_raw_power)]
                df = df[(df.freq >= args.freq_start) & (df.freq <= args.freq_end)]
                df = df[(now - df.ts).abs() < 60]
                df["scan_pos"] = (df.freq - args.freq_start) / (
                    args.freq_end - args.freq_start
                )
                if df.size:
                    lastfreq = df.freq.iat[-1]
                rotatelognow = False
                for row in df.itertuples():
                    rollover = row.scan_pos < 0.1 and max_scan_pos > 0.9 and fftbuffer
                    max_scan_pos = max(max_scan_pos, row.scan_pos)
                    if rollover:
                        max_scan_pos = row.scan_pos
                        frame_counter.inc()
                        new_lastbins = process_fft(
                            args,
                            prom_vars,
                            row.ts,
                            fftbuffer,
                            lastbins,
                            running_df,
                        )
                        if new_lastbins is not None:
                            lastbins = new_lastbins
                            if lastbins:
                                lastbins_history = [lastbins] + lastbins_history
                                lastbins_history = lastbins_history[: args.history]
                            call_record_signals(args, lastbins_history, prom_vars)
                        fftbuffer = []
                        rotate_age = now - openlogts
                        if rotate_age > args.rotatesecs:
                            rotatelognow = True
                        if runonce:
                            return
                    fftbuffer.append((row.ts, row.freq, row.pw))
                if rotatelognow:
                    break
        rotate_file_n(".".join((args.log, "zst")), args.nlog, require_initial=False)
        new_log = ".".join((args.log, "1"))
        os.rename(args.log, new_log)
        executor.submit(zstd_file, new_log)


def fft_proxy(args, buff_file, buffer_time=FFT_BUFFER_TIME, shutdown_str=None):
    zmq_addr = f"tcp://{args.logaddr}:{args.logport}"
    logging.info("connecting to %s", zmq_addr)
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(zmq_addr)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    packets_sent = 0
    last_packet_sent_time = time.time()
    tmp_buff_file = os.path.basename(buff_file)
    tmp_buff_file = buff_file.replace(tmp_buff_file, "." + tmp_buff_file)
    shutdown = False
    context = zstandard.ZstdCompressor()
    while not shutdown:
        with open(tmp_buff_file, "wb") as zbf:
            with context.stream_writer(zbf) as bf:
                while not shutdown:
                    sock_txt = socket.recv()
                    bf.write(sock_txt)
                    shutdown = (
                        shutdown_str is not None and sock_txt.find(shutdown_str) != -1
                    )
                    now = time.time()
                    if (
                        shutdown or now - last_packet_sent_time > buffer_time
                    ) and not os.path.exists(buff_file):
                        if packets_sent == 0:
                            logging.info("recording first FFT packet")
                        packets_sent += 1
                        last_packet_sent_time = now
                        break
        os.rename(tmp_buff_file, buff_file)


def find_signals(args, prom_vars):
    buff_file = os.path.join(args.buff_path, BUFF_FILE)
    with concurrent.futures.ProcessPoolExecutor(2) as executor:
        proxy_result = executor.submit(fft_proxy, args, buff_file)
        process_fft_lines(args, prom_vars, buff_file, executor, proxy_result)


def argument_parser():
    parser = argparse.ArgumentParser(
        description="watch a scan UDP stream and find signals"
    )
    parser.add_argument(
        "--log", default="scan.log", type=str, help="base path for scan logging"
    )
    parser.add_argument(
        "--fftlog",
        default="",
        type=str,
        help="if defined, path to log last complete FFT frame",
    )
    parser.add_argument(
        "--fftgraph",
        default="",
        type=str,
        help="if defined, path to write graph of most recent FFT and detected peaks",
    )
    parser.add_argument(
        "--nfftgraph", default=10, type=int, help="keep last N FFT graphs"
    )
    parser.add_argument(
        "--rotatesecs",
        default=3600,
        type=int,
        help="rotate scan log after this many seconds",
    )
    parser.add_argument(
        "--nlog", default=10, type=int, help="keep only this many scan.logs"
    )
    parser.add_argument(
        "--bin_mhz", default=20, type=int, help="monitoring bin width in MHz"
    )
    parser.add_argument(
        "--max_raw_power",
        default=50,
        type=float,
        help="maximum raw power permitted from FFT",
    )
    parser.add_argument(
        "--width",
        default=10,
        type=int,
        help=f"minimum signal width to detect a peak (multiple of {SCAN_FRES / 1e6} MHz, e.g. 10 is {10 * SCAN_FRES / 1e6} MHz)",
    )
    parser.add_argument(
        "--threshold",
        default=-35,
        type=float,
        help="minimum signal finding threshold (dB)",
    )
    parser.add_argument(
        "--prominence",
        default=2,
        type=float,
        help="minimum peak prominence (see scipy.signal.find_peaks)",
    )
    parser.add_argument(
        "--history",
        default=5,
        type=int,
        help="number of frames of signal history to keep",
    )
    parser.add_argument(
        "--recorder",
        action="append",
        default=[],
        help="SDR recorder base URLs (e.g. http://host:port/, multiples can be specified)",
    )
    parser.add_argument(
        "--record_bw_msps",
        default=20,
        type=int,
        help="record bandwidth in n * {MB} samples per second",
    )
    parser.add_argument(
        "--record_secs", default=10, type=int, help="record time duration in seconds"
    )
    parser.add_argument(
        "--promport",
        dest="promport",
        type=int,
        default=9000,
        help="Prometheus client port",
    )
    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        default=80,
        help="control webserver port",
    )
    parser.add_argument(
        "--freq-end",
        dest="freq_end",
        type=float,
        default=float(1e9),
        help="Set freq_end [default=%(default)r]",
    )
    parser.add_argument(
        "--freq-start",
        dest="freq_start",
        type=float,
        default=float(100e6),
        help="Set freq_start [default=%(default)r]",
    )
    parser.add_argument(
        "--logaddr",
        dest="logaddr",
        type=str,
        default="127.0.0.1",
        help="Log FFT results from this address",
    )
    parser.add_argument(
        "--logport",
        dest="logport",
        type=int,
        default=8001,
        help="Log FFT results from this port",
    )
    parser.add_argument(
        "--max_recorder_signals",
        dest="max_recorder_signals",
        type=int,
        default=1,
        help="Max number of recordings per worker to request",
    )
    parser.add_argument(
        "--running_fft_secs",
        dest="running_fft_secs",
        type=int,
        default=900,
        help="Number of seconds for running FFT average",
    )
    parser.add_argument(
        "--buff_path",
        dest="buff_path",
        type=str,
        default="/dev/shm",  # nosec
        help="Path for FFT buffer file",
    )
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")
    prom_vars = init_prom_vars()
    start_http_server(args.promport)
    x = threading.Thread(
        target=find_signals,
        args=(
            args,
            prom_vars,
        ),
    )
    x.start()
    app = falcon.App()
    scanner_form = ScannerForm()
    result = Result()
    active_requests = ActiveRequests()
    app.add_route("/", scanner_form)
    app.add_route("/result", result)
    app.add_route("/requests", active_requests)
    bjoern.run(app, "0.0.0.0", args.port)  # nosec
