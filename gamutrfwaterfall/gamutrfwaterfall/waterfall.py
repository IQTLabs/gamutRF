import datetime
import json
import logging
import multiprocessing
import os
import signal
import tempfile
import time
import warnings
import requests
import matplotlib
import zmq
from flask import (
    Flask,
    send_file,
    render_template,
    send_from_directory,
    request,
    redirect,
)
from matplotlib import style as matplotlibstyle

from gamutrflib.peak_finder import get_peak_finder
from gamutrflib.zmqbucket import ZmqReceiver, parse_scanners, frame_resample
from gamutrfwaterfall.argparser import argument_parser
from gamutrfwaterfall.waterfall_plot import reset_fig, init_fig, update_fig, make_config, WaterfallState

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


def waterfall(
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
    matplotlibstyle.use("fast")

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

            state = WaterfallState(config, base_save_path, peak_finder)
            matplotlib.use(config.engine)
            results = [
                (scan_configs, frame_resample(scan_df, config.freq_resolution * 1e6))
            ]

            if config_vars_path:
                get_scanner_args(api_endpoint, config_vars)
                write_scanner_args(config_vars_path, config_vars)

            need_reconfig = False
            need_init = True
        if need_init:
            init_fig(config, state, onresize)
            need_init = False
            need_reset_fig = True
        if need_reset_fig:
            reset_fig(config, state)
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
            update_fig(config, state, results)
            if config.batch and state.counter % config.reclose_interval == 0:
                need_init = True
            results = []
        else:
            time.sleep(0.1)
    zmqr.stop()


def get_scanner_args(api_endpoint, config_vars):
    try:
        response = requests.get(f"http://{api_endpoint}/getconf", timeout=30)
        response.raise_for_status()

        for name, val in json.loads(response.text).items():
            config_vars[name] = val

    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.HTTPError,
        requests.exceptions.ConnectTimeout,
    ) as err:
        logging.error(err)


def write_scanner_args(config_vars_path, config_vars):
    tmpfile = os.path.join(
        os.path.dirname(config_vars_path),
        "." + os.path.basename(config_vars_path),
    )
    with open(tmpfile, "w", encoding="utf8") as f:
        json.dump(config_vars, f)
    os.rename(tmpfile, config_vars_path)


class FlaskHandler:
    def __init__(
        self,
        savefig_path,
        tempdir,
        predictions,
        port,
        refresh,
        inference_server,
        inference_port,
        api_endpoint,
        config_vars,
        config_vars_path,
    ):
        self.inference_addr = f"tcp://{inference_server}:{inference_port}"
        self.savefig_path = savefig_path
        self.config_vars = config_vars
        self.config_vars_path = config_vars_path
        self.tempdir = tempdir
        self.predictions_file = "predictions.html"
        self.refresh = refresh
        self.predictions = predictions
        self.api_endpoint = api_endpoint
        self.app = Flask(__name__, template_folder="templates", static_folder="static")
        self.savefig_file = os.path.basename(self.savefig_path)
        self.app.add_url_rule("/", "index", self.serve_waterfall_page)
        self.app.add_url_rule(
            "/waterfall", "serve_waterfall_page", self.serve_waterfall_page
        )
        self.app.add_url_rule(
            "/waterfall_img", "serve_waterfall_img", self.serve_waterfall_img
        )
        self.app.add_url_rule(
            "/config_form", "config_form", self.config_form, methods=["POST", "GET"]
        )
        self.app.add_url_rule(
            "/predictions", "serve_predictions_page", self.serve_predictions_page
        )
        self.app.add_url_rule(
            "/predictions_content",
            "serve_predictions_content",
            self.serve_predictions_content,
        )
        self.app.add_url_rule("/<path:path>", "", self.serve)
        self.process = multiprocessing.Process(
            target=self.app.run,
            kwargs={"host": "0.0.0.0", "port": port},  # nosec
        )
        self.zmq_process = multiprocessing.Process(target=self.poll_zmq)
        self.write_predictions_content("no predictions yet")
        self.read_config_vars()

    def start(self):
        self.process.start()
        self.zmq_process.start()

    def write_predictions_content(self, content):
        tmpfile = os.path.join(self.tempdir, "." + self.predictions_file)
        with open(tmpfile, "w", encoding="utf8") as f:
            f.write(f"{content}")
        os.rename(tmpfile, os.path.join(self.tempdir, self.predictions_file))

    def poll_zmq(self):
        zmq_context = zmq.Context()
        socket = zmq_context.socket(zmq.SUB)
        socket.connect(self.inference_addr)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        DELIM = "\n\n"
        json_buffer = ""
        item_buffer = []

        while True:
            try:
                while True:
                    sock_txt = socket.recv(flags=zmq.NOBLOCK).decode("utf8")
                    json_buffer += sock_txt
            except zmq.error.Again:
                pass
            while True:
                delim_pos = json_buffer.find(DELIM)
                if delim_pos == -1:
                    break
                raw_item = json_buffer[:delim_pos]
                json_buffer = json_buffer[delim_pos + len(DELIM) :]
                try:
                    item = json.loads(raw_item)
                except json.decoder.JSONDecodeError:
                    continue
                ts = float(item["metadata"]["ts"])
                if "predictions_image_path" not in item["metadata"]:
                    continue
                ts = float(item["metadata"]["ts"])
                item_buffer.append((ts, item))
            item_buffer = item_buffer[-self.predictions :]
            predictions = sorted(item_buffer, key=lambda x: x[0], reverse=True)
            images = []
            now = time.time()
            for ts, item in predictions:
                image = item["metadata"]["predictions_image_path"]
                age = now - ts
                style = ""
                if age > 3 * self.refresh:
                    style = 'style="color:red;"'
                images.append(
                    "<p %s>%s (age %.1fs)</p><p %s><img src=%s></img></p>"
                    % (style, image, age, style, image)
                )
            if images:
                self.write_predictions_content(
                    f"<p>{datetime.datetime.now().isoformat()}</p>" + "".join(images)
                )
            time.sleep(0.1)

    def serve(self, path):
        if path:
            full_path = os.path.realpath(os.path.join("/", path))
            if os.path.exists(full_path):
                return send_file(full_path, mimetype="image/png")
            return "%s: not found" % full_path, 404
        if os.path.exists(self.savefig_path):
            return (
                '<html><head><meta http-equiv="refresh" content="%u"></head><body><img src="%s"></img></body></html>'
                % (self.refresh, self.savefig_file),
                200,
            )
        return (
            '<html><head><meta http-equiv="refresh" content="%u"></head><body>waterfall initializing, please wait or reload...</body></html>'
            % self.refresh,
            200,
        )

    def serve_predictions_content(self):
        return send_from_directory(self.tempdir, self.predictions_file)

    def serve_predictions_page(self):
        # return send_from_directory(self.tempdir, self.predictions_file)
        return render_template("predictions.html")

    def read_config_vars(self):
        try:
            with open(self.config_vars_path, encoding="utf8") as f:
                self.config_vars = json.load(f)
        except FileNotFoundError:
            pass

    def serve_waterfall_page(self):
        self.read_config_vars()
        return render_template("waterfall.html", config_vars=self.config_vars)

    # @app.route("/file/<path:filename>")
    # def serve_file(self, filename):
    #     return send_from_directory(self.tempdir, filename)

    def serve_waterfall_img(self):
        return send_from_directory(self.tempdir, self.savefig_file)

    def config_form(self):
        for var in self.config_vars:
            self.config_vars[var] = request.form.get(var, self.config_vars[var])
        write_scanner_args(self.config_vars_path, self.config_vars)
        reset = request.form.get("reset", None)
        if reset == "reset":
            reconf_query_str = "&".join(
                [f"{k}={v}" for k, v in self.config_vars.items()]
            )
            logging.info(f"\n\n{reconf_query_str=}\n\n")
            try:
                response = requests.get(
                    f"http://{self.api_endpoint}/reconf?{reconf_query_str}",
                    timeout=30,
                )
                logging.info(f"\n\n{response=}\n\n")
                logging.info(f"\n\n{response.text=}\n\n")
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                requests.exceptions.ConnectTimeout,
            ) as err:
                logging.error(str(err))
        return redirect("/", code=302)


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

        waterfall(
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
