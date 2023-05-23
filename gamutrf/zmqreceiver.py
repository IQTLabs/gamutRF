"""
Retrieve streaming scanner FFT results from a gr-iqtlabs/gamutRF scanner.

Example usage:

    from gamutrf.zmqreceiver import ZmqReceiver
    zmqr = ZmqReceiver("127.0.0.1", 3306)

    while True:
        scan_config, frame_df = zmqr.read_buffer()
        if frame_df is None:
            # no new scan result yet
            time.sleep(1)
        ...

    zmqr.stop()

scan_config and frame_df will be None if the next full scan has not been received yet,
and scan updates are automatically processed in the background to avoid ZMQ tranport
overflows.

frame_df is a pandas DataFrame with the results of a full scan, and scan_config
is a python dict containing scan metadata - the contents of the "config" dict, from
https://github.com/IQTLabs/gr-iqtlabs/blob/main/grc/iqtlabs_retune_fft.block.yml.
"""

import concurrent.futures
import tempfile
import json
import logging
import os
import pathlib
import time
import zmq
import zstandard
import pandas as pd

FFT_BUFFER_TIME = 1
BUFF_FILE = "scanfftbuffer.txt.zst"  # nosec


def fft_proxy(
    addr, port, buff_file, buffer_time=FFT_BUFFER_TIME, live_file=None, poll_timeout=1
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
    context = zstandard.ZstdCompressor()
    shutdown = False
    while not shutdown:
        with open(tmp_buff_file, "wb") as zbf:
            with context.stream_writer(zbf) as bf:
                while not shutdown:
                    shutdown = live_file is not None and not live_file.exists()
                    try:
                        sock_txt = socket.recv(flags=zmq.NOBLOCK)
                    except zmq.error.Again:
                        time.sleep(poll_timeout)
                        continue
                    bf.write(sock_txt)
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


class ZmqReceiver:
    def __init__(self, addr, port, buff_path=None, proxy=fft_proxy):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.live_file = pathlib.Path(os.path.join(self.tmpdir.name, "live_file"))
        self.live_file.touch()
        if buff_path is None:
            buff_path = self.tmpdir.name
        self.buff_file = os.path.join(buff_path, BUFF_FILE)
        self.context = zstandard.ZstdDecompressor()
        self.txt_buf = ""
        self.fftbuffer = None
        self.last_sweep_start = 0
        if os.path.exists(self.buff_file):
            os.remove(self.buff_file)
        self.executor = concurrent.futures.ProcessPoolExecutor(1)
        self.proxy_result = self.executor.submit(
            proxy, addr, port, self.buff_file, live_file=self.live_file
        )

    def stop(self):
        self.live_file.unlink()
        self.executor.shutdown()
        self.tmpdir.cleanup()

    def healthy(self):
        return os.path.exists(self.live_file) and self.proxy_result.running()

    def lines_to_df(self, lines):
        try:
            records = []
            for line in lines:
                line = line.strip()
                json_record = json.loads(line)
                ts = float(json_record["ts"])
                sweep_start = float(json_record["sweep_start"])
                buckets = json_record["buckets"]
                scan_config = json_record["config"]
                records.extend(
                    [
                        {
                            "ts": ts,
                            "freq": float(freq),
                            "db": float(db),
                            "sweep_start": sweep_start,
                        }
                        for freq, db in buckets.items()
                    ]
                )
            return (scan_config, pd.DataFrame(records))
        except ValueError as err:
            logging.error(str(err))
            return (None, None)

    def txtbuf_to_lines(self, log):
        lines = self.txt_buf.splitlines()
        if len(lines) > 1:
            if self.txt_buf.endswith("\n"):
                if log:
                    log.write(self.txt_buf)
                self.txt_buf = ""
            elif lines:
                last_line = lines[-1]
                if log:
                    log.write(self.txt_buf[: -len(last_line)])
                self.txt_buf = last_line
                lines = lines[:-1]
            return lines
        return None

    def read_new_frame_df(self, df):
        frame_df = None
        df = df[(time.time() - df.ts).abs() < 60]
        if df.size:
            lastfreq = df.freq.iat[-1]
            logging.info("last frequency read %f MHz", lastfreq / 1e6)
            max_sweep_start = df["sweep_start"].max()
            if max_sweep_start != self.last_sweep_start:
                if self.fftbuffer is None:
                    frame_df = df
                else:
                    frame_df = pd.concat(
                        [self.fftbuffer, df[df["sweep_start"] == self.last_sweep_start]]
                    )
                    self.fftbuffer = df[df["sweep_start"] != self.last_sweep_start]
                self.last_sweep_start = max_sweep_start
            else:
                if self.fftbuffer is None:
                    self.fftbuffer = df
                else:
                    self.fftbuffer = pd.concat([self.fftbuffer, df])
        return frame_df

    def read_buff(self, log=None):
        scan_config = None
        frame_df = None
        if os.path.exists(self.buff_file):
            logging.info("read %u bytes of FFT data", os.stat(self.buff_file).st_size)
            with self.context.stream_reader(open(self.buff_file, "rb")) as bf:
                self.txt_buf += bf.read().decode("utf8")
            os.remove(self.buff_file)
            lines = self.txtbuf_to_lines(log)
            if lines:
                scan_config, df = self.lines_to_df(lines)
                frame_df = self.read_new_frame_df(df)
        return (scan_config, frame_df)
