"""
Retrieve streaming scanner FFT results from a gr-iqtlabs/gamutRF scanner.

Example usage:

    from gamutrf.zmqreceiver import ZmqReceiver
    zmqr = ZmqReceiver(scanners=[("127.0.0.1", 8001)])

    while True:
        scan_configs, frame_df = zmqr.read_buffer()
        if frame_df is None:
            # no new scan result yet
            time.sleep(1)
        ...

    zmqr.stop()

scan_configs and frame_df will be None if the next full scan has not been received yet,
and scan updates are automatically processed in the background to avoid ZMQ tranport
overflows.

frame_df is a pandas DataFrame with the results of a full scan, and scan_configs is a
list of python dicts containing scan metadata - the contents of the "config" dict, from
https://github.com/IQTLabs/gr-iqtlabs/blob/main/grc/iqtlabs_retune_fft.block.yml.
"""

import concurrent.futures
import datetime
import tempfile
import json
import logging
import os
import pathlib
import re
import time
import zmq
import zstandard
import pandas as pd

FFT_BUFFER_TIME = 0.1
BUFF_FILE = "scanfftbuffer.txt.zst"  # nosec


def frame_resample(df, scan_fres):
    if df is not None:
        # ...first frequency
        df["freq"] = (df["freq"] / scan_fres).round() * scan_fres / 1e6
        df = df.set_index("freq")
        # ...then power
        df["db"] = df.groupby(["freq"])["db"].mean()
        df = df.reset_index().drop_duplicates(subset=["freq"])
        return df.sort_values("freq")
    return df


def parse_scanners(args_scanners):
    scanner_re = re.compile(r"^(.+):(\d+)$")
    scanners = []
    for scanner_str in args_scanners.split(","):
        scanner_match = scanner_re.match(scanner_str)
        if not scanner_match:
            raise ValueError(
                f"invalid scanner address: {scanner_str} from {args_scanners}"
            )
        scanners.append((scanner_match.group(1), int(scanner_match.group(2))))
    return scanners


def fft_proxy(
    addr, port, buff_file, buffer_time=FFT_BUFFER_TIME, live_file=None, poll_timeout=0.1
):
    zmq_addr = f"tcp://{addr}:{port}"
    logging.info("connecting to %s", zmq_addr)
    zmq_context = zmq.Context()
    socket = zmq_context.socket(zmq.SUB)
    socket.connect(zmq_addr)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    packets_sent = 0
    last_packet_sent_time = time.time()
    tmp_buff_file = os.path.basename(buff_file)
    tmp_buff_file = buff_file.replace(tmp_buff_file, "." + tmp_buff_file)
    if os.path.exists(tmp_buff_file):
        os.remove(tmp_buff_file)
    compress_context = zstandard.ZstdCompressor()
    decompress_context = zstandard.ZstdDecompressor()
    shutdown = False
    last_log_time = None
    last_data_time = None
    while not shutdown:
        with open(tmp_buff_file, "wb") as zbf:
            with compress_context.stream_writer(zbf) as bf:
                while not shutdown:
                    shutdown = live_file is not None and not live_file.exists()
                    now = time.time()
                    try:
                        sock_txt = socket.recv(flags=zmq.NOBLOCK)
                    except zmq.error.Again:
                        if last_log_time is None or now - last_log_time > 10:
                            if last_data_time is None:
                                logging.warning("no data yet from %s", zmq_addr)
                            else:
                                if now - last_data_time > 10:
                                    logging.warning(
                                        "no data from %s for %u seconds",
                                        zmq_addr,
                                        now - last_data_time,
                                    )
                            last_log_time = now
                        time.sleep(poll_timeout)
                        continue
                    # gamutrf might send compressed message
                    try:
                        sock_txt = decompress_context.decompress(sock_txt)
                    except zstandard.ZstdError:
                        pass
                    bf.write(sock_txt)
                    now = time.time()
                    last_data_time = now
                    if (
                        shutdown or now - last_packet_sent_time > buffer_time
                    ) and not os.path.exists(buff_file):
                        if packets_sent == 0:
                            logging.info("recording first FFT packet")
                        packets_sent += 1
                        last_packet_sent_time = now
                        break
        os.rename(tmp_buff_file, buff_file)


class ZmqScanner:
    def __init__(
        self,
        buff_path,
        proxy,
        addr,
        port,
        live_file,
        executor,
    ):
        self.buff_file = os.path.join(buff_path, "_".join((addr, str(port), BUFF_FILE)))
        if os.path.exists(self.buff_file):
            os.remove(self.buff_file)
        self.addr = addr
        self.port = port
        self.context = zstandard.ZstdDecompressor()
        self.fftbuffer = None
        self.scan_configs = {}
        self.proxy_result = executor.submit(
            proxy, addr, port, self.buff_file, live_file=live_file
        )

    def info(self, infostr):
        logging.info("%s:%u %s", self.addr, self.port, infostr)

    def healthy(self):
        return self.proxy_result.running()

    def __str__(self):
        return f"ZmqScanner on {self.addr}:{self.port}"

    def read_buff_file(self, log):
        lines = None
        if os.path.exists(self.buff_file):
            self.info("read %u bytes of FFT data" % os.stat(self.buff_file).st_size)
            with self.context.stream_reader(open(self.buff_file, "rb")) as bf:
                txt_buf = bf.read().decode("utf8")
                if log:
                    log.write(txt_buf)
                try:
                    lines = [json.loads(line) for line in txt_buf.splitlines() if line]
                except json.decoder.JSONDecodeError as err:
                    logging.info("%s: %s", err, txt_buf)
            os.remove(self.buff_file)
        return lines

    def read_new_frame_df(self, df, discard_time):
        frame_df = None
        scan_config = None
        if discard_time:
            df = df[(time.time() - df.ts).abs() < discard_time]
        if df.size:
            lastfreq = df["freq"].iat[-1]
            self.info("last frequency read %f MHz" % (lastfreq / 1e6))
            if self.fftbuffer is None:
                self.fftbuffer = df
            else:
                self.fftbuffer = pd.concat([self.fftbuffer, df])
            if self.fftbuffer["sweep_start"].nunique() > 1:
                min_sweep_start = self.fftbuffer["sweep_start"].min()
                max_sweep_start = self.fftbuffer["sweep_start"].max()
                frame_df = self.fftbuffer[
                    self.fftbuffer["sweep_start"] != max_sweep_start
                ].copy()
                frame_df["tune_count"] = (
                    frame_df["tune_count"].max() - frame_df["tune_count"].min()
                )
                self.fftbuffer = self.fftbuffer[
                    self.fftbuffer["sweep_start"] == max_sweep_start
                ]
                scan_config = self.scan_configs[min_sweep_start]
                del self.scan_configs[min_sweep_start]
        return (scan_config, frame_df)

    def lines_to_df(self, lines):
        try:
            records = []
            for json_record in lines:
                ts = float(json_record["ts"])
                sweep_start = float(json_record["sweep_start"])
                total_tune_count = int(json_record["total_tune_count"])
                buckets = json_record["buckets"]
                scan_config = json_record["config"]
                self.scan_configs[sweep_start] = scan_config
                records.extend(
                    [
                        {
                            "ts": ts,
                            "freq": float(freq),
                            "db": float(db),
                            "sweep_start": sweep_start,
                            "tune_count": total_tune_count,
                        }
                        for freq, db in buckets.items()
                    ]
                )
            return pd.DataFrame(records)
        except ValueError as err:
            logging.error(str(err))
            return None

    def read_buff(self, log, discard_time):
        scan_config = None
        frame_df = None
        lines = self.read_buff_file(log)
        if lines:
            df = self.lines_to_df(lines)
            if df is not None:
                scan_config, frame_df = self.read_new_frame_df(df, discard_time)
        return scan_config, frame_df


class ZmqReceiver:
    def __init__(
        self,
        scanners=[("127.0.0.1", 8001)],
        buff_path=None,
        proxy=fft_proxy,
    ):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.live_file = pathlib.Path(os.path.join(self.tmpdir.name, "live_file"))
        self.live_file.touch()
        if buff_path is None:
            buff_path = self.tmpdir.name
        self.executor = concurrent.futures.ProcessPoolExecutor(len(scanners))
        self.scanners = []
        self.last_results = []
        for addr, port in scanners:
            self.scanners.append(
                ZmqScanner(buff_path, proxy, addr, port, self.live_file, self.executor)
            )

    def stop(self):
        self.live_file.unlink()
        self.executor.shutdown()
        self.tmpdir.cleanup()

    def healthy(self):
        if os.path.exists(self.live_file):
            for scanner in self.scanners:
                if not scanner.healthy():
                    return False
            return True
        return False

    def read_buff(self, log=None, discard_time=0, scan_fres=0):
        while True:
            results = [
                scanner.read_buff(log, discard_time) for scanner in self.scanners
            ]
            new_results = 0
            if self.last_results:
                for i, result in enumerate(results):
                    _scan_config, df = result
                    if df is not None:
                        self.last_results[i] = result
                        new_results += 1
                        logging.info("%s got scan result", self.scanners[i])
            else:
                self.last_results = results
            if new_results == 0:
                break

        scan_configs = []
        dfs = []

        for scan_config, df in self.last_results:
            scan_configs.append(scan_config)
            if df is not None:
                dfs.append(df)

        df = None
        if len(dfs) == len(self.scanners):
            df = pd.concat(dfs)
            logging.info(
                "all scanners got result, %s to %s",
                datetime.datetime.fromtimestamp(df.ts.min()).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                datetime.datetime.fromtimestamp(df.ts.max()).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            )
            if scan_fres:
                df = frame_resample(df, scan_fres)
            self.last_results = []

        return (scan_configs, df)
