import logging
import signal
import sys
from argparse import ArgumentParser

try:
    from gnuradio import iqtlabs  # pytype: disable=import-error
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
        default=eng_notation.num_to_str(float(4.096e6)),
        help="Set samp_rate [default=%(default)r]",
    )
    parser.add_argument(
        "--sweep-sec",
        dest="sweep_sec",
        type=float,
        default=30,
        help="Set sweep_sec [default=%(default)r]",
    )
    parser.add_argument(
        "--tune-step-fft",
        dest="tune_step_fft",
        type=int,
        default=0,
        help="tune FFT step (0 is use sweep_sec) [default=%(default)r]",
    )
    parser.add_argument(
        "--skip-tune-step",
        dest="skip_tune_step",
        type=int,
        default=0,
        help="skip FFT samples on retune [default=%(default)r]",
    )
    parser.add_argument(
        "--sample_dir",
        dest="sample_dir",
        type=str,
        default="",
        help="where to write samples/FFT points",
    )
    parser.add_argument(
        "--write_samples",
        dest="write_samples",
        default=0,
        type=int,
        help="if > 0, write FFT/raw samples to --sample_dir",
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
        "--tuneoverlap",
        dest="tuneoverlap",
        type=float,
        default=0.5,
        help="multiple of samp_rate when retuning",
    )
    parser.add_argument(
        "--bucket_range",
        dest="bucket_range",
        type=float,
        default=1.0,
        help="what proportion of FFT buckets to use",
    )
    parser.add_argument(
        "--inference_plan_file",
        dest="inference_plan_file",
        type=str,
        default="",
        help="full path to plan file for wavelearner",
    )
    parser.add_argument(
        "--inference_output_dir",
        dest="inference_output_dir",
        type=str,
        default="",
        help="directory for inference output",
    )
    parser.add_argument(
        "--inference_input_len",
        dest="inference_input_len",
        type=int,
        default=2048,
        help="vector length for wavelearner",
    )
    parser.add_argument(
        "--tuning_ranges",
        dest="tuning_ranges",
        type=str,
        default="",
        help="tuning ranges (overriding freq_start and freq_end)",
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

    # ensure tuning tags arrive on FFT window boundaries.
    if options.samp_rate % options.nfft:
        print("NFFT should be a factor of sample rate")

    if options.write_samples and not options.sample_dir:
        print("Must provide --sample_dir when writing samples/points")
        sys.exit(1)

    wavelearner = None
    try:
        import wavelearner as wavelearner_lib  # pytype: disable=import-error

        wavelearner = wavelearner_lib
        print("using wavelearner")
    except ModuleNotFoundError:
        print("wavelearner not available")

    prom_vars = init_prom_vars()
    prom_vars["freq_start_hz"].set(options.freq_start)
    prom_vars["freq_end_hz"].set(options.freq_end)
    prom_vars["sweep_sec"].set(options.sweep_sec)
    start_http_server(options.promport)

    tb = grscan(
        freq_end=options.freq_end,
        freq_start=options.freq_start,
        igain=options.igain,
        samp_rate=options.samp_rate,
        sweep_sec=options.sweep_sec,
        logaddr=options.logaddr,
        logport=options.logport,
        sdr=options.sdr,
        sdrargs=options.sdrargs,
        fft_size=options.nfft,
        tune_overlap=options.tuneoverlap,
        tune_step_fft=options.tune_step_fft,
        skip_tune_step=options.skip_tune_step,
        sample_dir=options.sample_dir,
        write_samples=options.write_samples,
        bucket_range=options.bucket_range,
        tuning_ranges=options.tuning_ranges,
        inference_plan_file=options.inference_plan_file,
        inference_output_dir=options.inference_output_dir,
        inference_input_len=options.inference_input_len,
        iqtlabs=iqtlabs,
        wavelearner=wavelearner,
    )

    def sig_handler(_sig=None, _frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.wait()
