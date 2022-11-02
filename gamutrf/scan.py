import logging
import signal
import sys
import time
from argparse import ArgumentParser

try:
    import habets39  # pytype: disable=import-error
    from gnuradio import eng_notation  # pytype: disable=import-error
    from gnuradio import gr  # pytype: disable=import-error
    from gnuradio.eng_arg import eng_float  # pytype: disable=import-error
    from gnuradio.eng_arg import intx  # pytype: disable=import-error
except ModuleNotFoundError:  # pragma: no cover
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme)"
    )
    sys.exit(1)

from prometheus_client import Gauge
from prometheus_client import start_http_server

from gamutrf.grscan import grscan


def init_prom_vars():
    prom_vars = {
        "freq_start_hz": Gauge("freq_start_hz", "start of scanning range in Hz"),
        "freq_end_hz": Gauge("freq_end_hz", "end of scanning range in Hz"),
        "sweep_sec": Gauge("sweep_sec", "scan sweep rate in seconds"),
        "retune_hz": Gauge("retune_hz", "retune rate per second"),
        "last_freq_update": Gauge("last_freq_update", "last frequency update time"),
    }
    return prom_vars


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--freq-end",
        dest="freq_end",
        type=eng_float,
        default=eng_notation.num_to_str(float(1e9)),
        help="Set freq_end [default=%(default)r]",
    )
    parser.add_argument(
        "--freq-start",
        dest="freq_start",
        type=eng_float,
        default=eng_notation.num_to_str(float(100e6)),
        help="Set freq_start [default=%(default)r]",
    )
    parser.add_argument(
        "--igain",
        dest="igain",
        type=intx,
        default=0,
        help="Set igain[default=%(default)r]",
    )
    parser.add_argument(
        "--samp-rate",
        dest="samp_rate",
        type=eng_float,
        default=eng_notation.num_to_str(float(4e6)),
        help="Set samp_rate [default=%(default)r]",
    )
    parser.add_argument(
        "--sweep-sec",
        dest="sweep_sec",
        type=intx,
        default=30,
        help="Set sweep_sec [default=%(default)r]",
    )
    parser.add_argument(
        "--retune-hz",
        dest="retune_hz",
        type=intx,
        default=97,
        help="Set retune rate per second [default=%(default)r]",
    )
    parser.add_argument(
        "--nfft",
        dest="nfft",
        type=int,
        default=2048,
        help="FFTI size [default=%(default)r]",
    )
    parser.add_argument(
        "--logaddr",
        dest="logaddr",
        type=str,
        default="0.0.0.0",  # nosec
        help="Log FFT results to this address",
    )
    parser.add_argument(
        "--logport",
        dest="logport",
        type=int,
        default=8001,
        help="Log FFT results to this port",
    )
    parser.add_argument(
        "--promport",
        dest="promport",
        type=int,
        default=9000,
        help="Prometheus client port",
    )
    parser.add_argument(
        "--sdr",
        dest="sdr",
        type=str,
        default="ettus",
        help="SDR to use (ettus, bladerf, or lime)",
    )
    parser.add_argument(
        "--sdrargs",
        dest="sdrargs",
        type=str,
        default="",
        help="extra args to pass to SDR driver",
    )
    parser.add_argument(
        "--updatetimeout",
        dest="updatetimeout",
        type=int,
        default=10,
        help="seconds to wait for healthy freq updates",
    )
    parser.add_argument(
        "--retune-intervals",
        dest="retune_intervals",
        type=int,
        default=1,
        help="if > 1, then schedule future retuning events in batches",
    )
    return parser


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")
    options = argument_parser().parse_args()
    if gr.enable_realtime_scheduling() != gr.RT_OK:
        print("Warning: failed to enable real-time scheduling.")

    if options.freq_start > options.freq_end:
        print("Error: freq_start must be less than freq_end")
        sys.exit(1)

    if options.freq_end > 6e9:
        print("Error: freq_end must be less than 6GHz")
        sys.exit(1)

    if options.freq_start < 70e6:
        print("Error: freq_start must be at least 70MHz")
        sys.exit(1)

    prom_vars = init_prom_vars()
    prom_vars["freq_start_hz"].set(options.freq_start)
    prom_vars["freq_end_hz"].set(options.freq_end)
    prom_vars["sweep_sec"].set(options.sweep_sec)
    prom_vars["retune_hz"].set(options.retune_hz)
    start_http_server(options.promport)

    tb = grscan(
        freq_end=options.freq_end,
        freq_start=options.freq_start,
        igain=options.igain,
        samp_rate=options.samp_rate,
        sweep_sec=options.sweep_sec,
        retune_hz=options.retune_hz,
        logaddr=options.logaddr,
        logport=options.logport,
        sdr=options.sdr,
        sdrargs=options.sdrargs,
        fft_size=options.nfft,
        retune_intervals=options.retune_intervals,
        habets39=habets39,
    )

    def sig_handler(_sig=None, _frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    while True:
        prom_vars["last_freq_update"].set(tb.freq_update)
        sleep_time = 1
        time.sleep(sleep_time)
        if not tb.freq_updated(options.updatetimeout):
            tb.stop()
            break

    tb.wait()
