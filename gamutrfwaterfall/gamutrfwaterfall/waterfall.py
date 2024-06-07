import logging
import os
import signal
import tempfile
import time
import warnings

from gamutrflib.peak_finder import get_peak_finder
from gamutrflib.zmqbucket import ZmqReceiver, parse_scanners, frame_resample
from gamutrfwaterfall.argparser import argument_parser
from gamutrfwaterfall.flask_handler import (
    FlaskHandler,
    get_scanner_args,
    write_scanner_args,
)
from gamutrfwaterfall.waterfall_plot import (
    make_config,
    WaterfallPlot,
)

warnings.filterwarnings(action="ignore", message="Mean of empty slice")
warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
warnings.filterwarnings(action="ignore", message="Degrees of freedom <= 0 for slice.")

SCAN_FRES = 1e4

CONFIG_VARS = {
    "freq_start": None,
    "freq_end": None,
    "igain": None,
    "tuneoverlap": None,
    "tune_step_fft": None,
    "sweep_sec": None,
}


def serve_waterfall(
    min_freq,
    max_freq,
    plot_snr,
    top_n,
    base_save_path,
    save_time,
    peak_finder,
    engine,
    savefig_path,
    rotate_secs,
    width,
    height,
    waterfall_height,
    waterfall_width,
    batch,
    refresh,
    zmqr,
    api_endpoint,
    config_vars,
    config_vars_path,
):
    global need_reset_fig
    need_reset_fig = True
    global running
    running = True
    need_init = True
    need_reconfig = True

    def onresize(_event):  # nosemgrep
        global need_reset_fig
        need_reset_fig = True

    def sig_handler(_sig=None, _frame=None):
        global running
        running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    logging.info("awaiting scanner startup")
    while not zmqr.healthy():
        time.sleep(0.1)

    logging.info("awaiting initial config from scanner(s)")
    scan_configs = None
    scan_df = None
    while zmqr.healthy() and running:
        scan_configs, scan_df = zmqr.read_buff(scan_fres=SCAN_FRES)
        if scan_df is not None:
            break
        time.sleep(0.1)

    if not scan_configs:
        return

    plot = None

    while zmqr.healthy() and running:
        if need_reconfig:
            config = make_config(
                scan_configs,
                min_freq,
                max_freq,
                engine,
                plot_snr,
                savefig_path,
                top_n,
                base_save_path,
                width,
                height,
                waterfall_height,
                waterfall_width,
                batch,
                rotate_secs,
                save_time,
            )
            logging.info(
                "scanning %fMHz to %fMHz at %fMsps with %u FFT points at %fMHz resolution",
                config.min_freq,
                config.max_freq,
                config.sampling_rate / 1e6,
                config.fft_len,
                config.freq_resolution,
            )
            plot = WaterfallPlot(config, base_save_path, peak_finder, 1)
            results = [
                (scan_configs, frame_resample(scan_df, config.freq_resolution * 1e6))
            ]

            if config_vars_path:
                get_scanner_args(api_endpoint, config_vars)
                write_scanner_args(config_vars_path, config_vars)

            need_reconfig = False
            need_init = True
        if need_init:
            plot.init_fig(onresize)
            need_init = False
            need_reset_fig = True
        if need_reset_fig:
            plot.reset_fig()
            need_reset_fig = False
        last_gap = time.time()
        while True:
            gap_time = time.time() - last_gap
            if results and gap_time > refresh / 2:
                break
            scan_configs, scan_df = zmqr.read_buff(
                scan_fres=config.freq_resolution * 1e6
            )
            if scan_df is None:
                if batch and gap_time < refresh / 2:
                    time.sleep(0.1)
                    continue
                break
            last_config = make_config(
                scan_configs,
                min_freq,
                max_freq,
                engine,
                plot_snr,
                savefig_path,
                top_n,
                base_save_path,
                width,
                height,
                waterfall_height,
                waterfall_width,
                batch,
                rotate_secs,
                save_time,
            )
            if last_config != config:
                logging.info("scanner config change detected")
                results = []
                need_reconfig = True
                break
            scan_df = scan_df[
                (scan_df.freq >= config.min_freq) & (scan_df.freq <= config.max_freq)
            ]
            if scan_df.empty:
                logging.info(
                    f"Scan is outside specified frequency range ({config.min_freq} to {config.max_freq})."
                )
                continue
            results.append((scan_configs, scan_df))
        if need_reconfig:
            continue
        if results:
            plot.update_fig(results)
            if plot.need_init():
                need_init = True
            results = []
        else:
            time.sleep(0.1)
    zmqr.stop()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argument_parser()
    args = parser.parse_args()
    detection_type = args.detection_type.lower()
    peak_finder = get_peak_finder(detection_type)

    if (detection_type and not args.n_detect) or (not detection_type and args.n_detect):
        raise ValueError("--detection_type and --n_detect must be set together")

    with tempfile.TemporaryDirectory() as tempdir:
        flask = None
        savefig_path = None
        engine = "GTK3Agg"
        batch = False
        config_vars_path = None
        config_vars = CONFIG_VARS

        if args.port:
            engine = "agg"
            batch = True
            savefig_path = os.path.join(tempdir, "waterfall.png")
            config_vars_path = os.path.join(tempdir, "config_vars.json")
            flask = FlaskHandler(
                savefig_path,
                tempdir,
                args.predictions,
                args.port,
                args.refresh,
                args.inference_server,
                args.inference_port,
                args.api_endpoint,
                config_vars,
                config_vars_path,
            )
            flask.start()

        zmqr = ZmqReceiver(
            scanners=parse_scanners(args.scanners),
        )

        serve_waterfall(
            args.min_freq,
            args.max_freq,
            args.plot_snr,
            args.n_detect,
            args.save_path,
            args.save_time,
            peak_finder,
            engine,
            savefig_path,
            args.rotate_secs,
            args.width,
            args.height,
            args.waterfall_height,
            args.waterfall_width,
            batch,
            args.refresh,
            zmqr,
            args.api_endpoint,
            config_vars,
            config_vars_path,
        )


if __name__ == "__main__":
    main()
