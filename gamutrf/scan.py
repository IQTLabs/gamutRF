import logging
import signal
import time
import sys
from argparse import ArgumentParser

try:
    from gnuradio import iqtlabs  # pytype: disable=import-error
    from gnuradio import eng_notation  # pytype: disable=import-error
    from gnuradio import gr  # pytype: disable=import-error
    from gnuradio.eng_arg import eng_float  # pytype: disable=import-error
    from gnuradio.eng_arg import intx  # pytype: disable=import-error
except ModuleNotFoundError as err:  # pragma: no cover
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme): %s"
        % err
    )
    sys.exit(1)

from prometheus_client import Gauge
from prometheus_client import start_http_server

from gamutrf.grscan import grscan
from gamutrf.flask_handler import FlaskHandler

running = True


def init_prom_vars():
    prom_vars = {
        "freq_start_hz": Gauge("freq_start_hz", "start of scanning range in Hz"),
        "freq_end_hz": Gauge("freq_end_hz", "end of scanning range in Hz"),
        "igain": Gauge("igain", "input gain"),
        "tune_overlap": Gauge("tune_overlap", "multiple of samp_rate when retuning"),
        "tune_step_fft": Gauge("tune_step_fft", "tune FFT step (0 is use sweep_sec)"),
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
        "--apiport",
        dest="apiport",
        type=int,
        default=9001,
        help="API server port",
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
    parser.add_argument(
        "--description",
        dest="description",
        type=str,
        default="",
        help="optional text description to provide with scanner updates",
    )
    parser.add_argument(
        "--scaling",
        dest="scaling",
        type=str,
        default="spectrum",
        help="""Same as --scaling parameter in scipy.signal.spectrogram(). 
        Selects between computing the power spectral density ('density') 
        where `Sxx` has units of V**2/Hz and computing the power
        spectrum ('spectrum') where `Sxx` has units of V**2, if `x`
        is measured in V and `fs` is measured in Hz. Defaults to
        'spectrum'.""",
    )
    return parser


def check_options(options):
    if options.samp_rate % options.nfft:
        print("NFFT should be a factor of sample rate")

    if options.freq_start > options.freq_end:
        return "freq_start must be less than freq_end"

    if options.freq_end > 6e9:
        return "freq_end must be less than 6GHz"

    if options.freq_start < 70e6:
        return "freq_start must be at least 70MHz"

    if options.write_samples and not options.sample_dir:
        return "Must provide --sample_dir when writing samples/points"

    if options.scaling not in ["spectrum", "density"]:
        return "scaling must be 'spectrum' or 'density'"

    return ""


def sig_handler(_sig=None, _frame=None):
    global running
    running = False


def run_loop(options, prom_vars, wavelearner):
    reconfigures = 0
    global running
    running = True

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    handler = FlaskHandler(options, check_options, ["promport", "apiport"])
    handler.start()

    while running:
        prom_vars["freq_start_hz"].set(handler.options.freq_start)
        prom_vars["freq_end_hz"].set(handler.options.freq_end)
        prom_vars["igain"].set(handler.options.igain)
        prom_vars["sweep_sec"].set(handler.options.sweep_sec)
        prom_vars["tune_overlap"].set(handler.options.tuneoverlap)
        prom_vars["tune_step_fft"].set(handler.options.tune_step_fft)
        tb = grscan(
            freq_end=handler.options.freq_end,
            freq_start=handler.options.freq_start,
            tuning_ranges=handler.options.tuning_ranges,
            igain=handler.options.igain,
            sweep_sec=handler.options.sweep_sec,
            tune_overlap=handler.options.tuneoverlap,
            tune_step_fft=handler.options.tune_step_fft,
            samp_rate=handler.options.samp_rate,
            logaddr=handler.options.logaddr,
            logport=handler.options.logport,
            sdr=handler.options.sdr,
            sdrargs=handler.options.sdrargs,
            fft_size=handler.options.nfft,
            skip_tune_step=handler.options.skip_tune_step,
            sample_dir=handler.options.sample_dir,
            write_samples=handler.options.write_samples,
            bucket_range=handler.options.bucket_range,
            inference_plan_file=handler.options.inference_plan_file,
            inference_output_dir=handler.options.inference_output_dir,
            inference_input_len=handler.options.inference_input_len,
            scaling=handler.options.scaling,
            description=handler.options.description,
            iqtlabs=iqtlabs,
            wavelearner=wavelearner,
        )
        tb.start()
        while running and reconfigures == handler.reconfigures:
            idle_time = 1
            time.sleep(idle_time)

        while reconfigures != handler.reconfigures:
            reconfigures = handler.reconfigures

        tb.stop()
        tb.wait()
        del tb


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")
    options = argument_parser().parse_args()
    if gr.enable_realtime_scheduling() != gr.RT_OK:
        print("Warning: failed to enable real-time scheduling.")
    results = check_options(options)
    if results:
        print(results)
        sys.exit(1)

    wavelearner = None
    try:
        import wavelearner as wavelearner_lib  # pytype: disable=import-error

        wavelearner = wavelearner_lib
        print("using wavelearner")
    except ModuleNotFoundError:
        print("wavelearner not available")

    prom_vars = init_prom_vars()
    start_http_server(options.promport)

    run_loop(options, prom_vars, wavelearner)
