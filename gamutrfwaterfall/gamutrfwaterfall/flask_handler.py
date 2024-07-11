import datetime
import json
import logging
import multiprocessing
import os
import time
import requests
import zmq
from flask import (
    Flask,
    send_file,
    render_template,
    send_from_directory,
    request,
    redirect,
)


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

    def serve_png(self, png_file):
        full_path = os.path.realpath(os.path.join(self.tempdir, png_file))
        if os.path.exists(full_path):
            return send_file(full_path, mimetype="image/png")
        return "%s: not found" % full_path, 404

    def serve_meta(self, png_file):
        full_path = os.path.realpath(os.path.join(self.tempdir, png_file))
        if os.path.exists(full_path):
            return (
                '<html><head><meta http-equiv="refresh" content="%u"></head><body><img src="%s"></img></body></html>'
                % (self.refresh, png_file),
                200,
            )
        return (
            '<html><head><meta http-equiv="refresh" content="%u"></head><body>waterfall initializing, please wait or reload...</body></html>'
            % self.refresh,
            200,
        )

    def serve(self, path):
        if path.endswith(".png"):
            return self.serve_png(os.path.basename(path))

        try:
            waterfall_n = int(path)
            if waterfall_n:
                return self.serve_meta(f"{waterfall_n}-waterfall.png")
        except ValueError:
            pass

        if not path:
            return self.serve_meta(os.path.basename(self.savefig_path))
        return "%s: not found" % path

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
            reconf_queries = []
            for k, v in self.config_vars.items():
                if k in ["description"]:
                    reconf_queries.append(f'{k}="{v}"')
                else:
                    reconf_queries.append(f"{k}={v}")
            reconf_query_str = "&".join(reconf_queries)
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
