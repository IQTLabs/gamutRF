import copy
import threading
from flask import Flask, request, render_template
import re
import requests


class FlaskHandler:
    def __init__(self, options, check_options, banned_args):
        self.check_options = check_options
        self.options = options
        self.banned_args = banned_args
        self.reconfigures = 0
        self.app = Flask(__name__)
        self.app.add_url_rule("/reconf", "reconf", self.reconf)
        self.app.add_url_rule("/index", "index", self.index)
        self.request = request
        self.thread = threading.Thread(
            target=self.app.run,
            kwargs={"host": "0.0.0.0", "port": options.apiport},  # nosec
            daemon=True,
        )

    def start(self):
        self.thread.start()

    def index(self):
        # TODO: Make these "basic" vs. "advanced" variables configurable
        basic_var_names = ["freq_start", "freq_end", "samp_rate", "igain"]
        advanced_var_names = [
            var_name
            for var_name in list(vars(self.options).keys())
            if var_name not in basic_var_names
        ]

        # TODO: Make these locations and ports configurable
        waterfall_location = "http://localhost:9003/waterfall.png"  # nosemgrep
        birdseye_location = "http://localhost:4999"  # nosemgrep

        birdseye_data = None  # nosemgrep

        birdseye_req = requests.get(birdseye_location)  # nosemgrep

        if birdseye_req.status_code != 200:
            birdseye_data = "No data available"
        else:
            birdseye_data = birdseye_req.content
            match = re.search(r'src="(.*?)"', birdseye_data.decode("utf8"))
            birdseye_data = match.group(1)

        return render_template(
            "index.html",
            basic_data=dict((key, self.options(key)) for key in basic_var_names),
            advanced_data=dict((key, self.options(key)) for key in advanced_var_names),
            waterfall_location=waterfall_location,
            birdseye_data=birdseye_data,
        )

    def reconf(self):
        new_options = copy.deepcopy(self.options)
        for arg, val in self.request.args.items():
            if arg in self.banned_args:
                continue
            if not hasattr(new_options, arg):
                return f"no such option {arg}", 400
            val_type = type(getattr(self.options, arg))
            try:
                setattr(new_options, arg, val_type(val))
            except (TypeError, ValueError) as err:
                return f"cannot set {arg} = {val}: {err}", 400
        results = self.check_options(new_options)
        if results:
            return results, 400
        self.options = new_options
        self.reconfigures += 1
        return "reconf", 200
