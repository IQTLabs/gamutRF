import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import threading
import time

import bjoern
import falcon
import jinja2
import numpy as np
import requests
import schedule

from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import start_http_server

from gamutrf.sigwindows import calc_db
from gamutrf.sigwindows import choose_record_signal
from gamutrf.sigwindows import choose_recorders
from gamutrf.sigwindows import get_center
from gamutrf.sigwindows import parse_freq_excluded
from gamutrf.sigwindows import find_sig_windows
from gamutrf.sigwindows import ROLLING_FACTOR
from gamutrf.utils import rotate_file_n, SCAN_FRES
from gamutrf.zmqreceiver import ZmqReceiver, parse_scanners


MB = int(1.024e6)
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
            action = req.media["action"]
            recorder_args = f"{action}/{signal_hz}/{record_samples}/{record_bps}"
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


def process_scan(args, scan_configs, prom_vars, df, lastbins):
    global PEAK_DBS
    df = calc_db(df, args.db_rolling_factor)
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
        tmp_fftlog = os.path.join(
            os.path.dirname(args.fftlog), "." + os.path.basename(args.fftlog)
        )
        df.to_csv(tmp_fftlog, sep="\t", header=False, index=False)
        os.rename(tmp_fftlog, args.fftlog)
    monitor_bins = set()
    peak_dbs = {}
    bin_freq_count = prom_vars["bin_freq_count"]
    last_bin_freq_time = prom_vars["last_bin_freq_time"]
    freq_start_mhz = (
        min([scan_config["freq_start"] for scan_config in scan_configs]) / 1e6
    )
    signals = []
    for scan_config in scan_configs:
        scan_df = df[
            (df.freq >= scan_config["freq_start"] / 1e6)
            & (df.freq <= scan_config["freq_end"] / 1e6)
        ]
        logging.info(args.detection_type)
        if args.detection_type:
            signals.extend(
                find_sig_windows(
                    scan_df,
                    args.detection_type,
                )
            )
    min_samp_rate = (
        min([scan_config["sample_rate"] for scan_config in scan_configs]) / 1e6
    )
    df.loc[freqdiffs > min_samp_rate * 2, "db"] = np.nan

    if PEAK_TRIGGER == 1 and signals:
        led_sleep = 0.2
        GPIO.output(PIN_TRIGGER, GPIO.HIGH)
        time.sleep(led_sleep)
        GPIO.output(PIN_TRIGGER, GPIO.LOW)

    ts = df["ts"].max()
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


class ScanProcessor:
    def __init__(self, args, prom_vars):
        self.args = args
        self.prom_vars = prom_vars
        self.lastbins_history = []
        self.lastbins = set()
        self.frame_counter = self.prom_vars["frame_counter"]


def process_scans(args, prom_vars, executor, zmqr):
    sp = ScanProcessor(args, prom_vars)

    while True:
        if os.path.exists(args.log):
            logging.info(f"{args.log} exists, will append first")
            mode = "a"
        else:
            logging.info(f"opening {args.log}")
            mode = "w"
        deadline = int(time.time()) + args.rotatesecs
        with open(args.log, mode=mode, encoding="utf-8") as l:
            while True:
                now = int(time.time())
                if not zmqr.healthy():
                    return
                if now > deadline:
                    break
                scan_configs, frame_df = zmqr.read_buff(l, scan_fres=SCAN_FRES)
                if frame_df is None:
                    schedule.run_pending()
                    sleep_time = 1
                    time.sleep(sleep_time)
                    continue
                sp.frame_counter.inc()
                logging.info(
                    "frame with sweep_start %us ago",
                    now - frame_df["sweep_start"].min(),
                )
                new_lastbins = process_scan(
                    args,
                    scan_configs,
                    prom_vars,
                    frame_df,
                    sp.lastbins,
                )
                if new_lastbins is not None:
                    sp.lastbins = new_lastbins
                    if sp.lastbins:
                        sp.lastbins_history = [sp.lastbins] + sp.lastbins_history
                        sp.lastbins_history = sp.lastbins_history[: args.history]
                call_record_signals(args, sp.lastbins_history, prom_vars)
        rotate_file_n(".".join((args.log, "zst")), args.nlog, require_initial=False)
        new_log = ".".join((args.log, "1"))
        os.rename(args.log, new_log)
        executor.submit(zstd_file, new_log)


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
        "--scanners",
        dest="scanners",
        type=str,
        default="127.0.0.1:8001",
        help="Connect to gamutRF scanners at these addresses",
    )
    parser.add_argument(
        "--max_recorder_signals",
        dest="max_recorder_signals",
        type=int,
        default=1,
        help="Max number of recordings per worker to request",
    )
    parser.add_argument(
        "--buff_path",
        dest="buff_path",
        type=str,
        default="/dev/shm",  # nosec
        help="Path for FFT buffer file",
    )
    parser.add_argument(
        "--db_rolling_factor",
        dest="db_rolling_factor",
        type=float,
        default=0,
        help="Divisor for rolling dB average (or 0 to disable)",
    )
    parser.add_argument(
        "--detection_type",
        default="narrowband",
        type=str,
        help="Detection type to plot (wideband, narrowband).",
    )
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    scanners = parse_scanners(args.scanners)

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")
    prom_vars = init_prom_vars()
    app = falcon.App()
    scanner_form = ScannerForm()
    result = Result()
    active_requests = ActiveRequests()
    app.add_route("/", scanner_form)
    app.add_route("/result", result)
    app.add_route("/requests", active_requests)
    start_http_server(args.promport)

    with concurrent.futures.ProcessPoolExecutor(2) as executor:
        zmqr = ZmqReceiver(
            scanners=scanners,
            buff_path=args.buff_path,
        )
        x = threading.Thread(
            target=process_scans,
            args=(
                args,
                prom_vars,
                executor,
                zmqr,
            ),
        )
        x.start()
        try:
            bjoern.run(app, "0.0.0.0", args.port)  # nosec
        except KeyboardInterrupt:
            logging.info("interrupt!")
            zmqr.stop()
            executor.shutdown()
            x.join()
