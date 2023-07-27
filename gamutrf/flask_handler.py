import copy
import threading
from flask import Flask, request


class FlaskHandler:
    def __init__(self, options, check_options, banned_args):
        self.check_options = check_options
        self.options = options
        self.banned_args = banned_args
        self.reconfigures = 0
        self.app = Flask(__name__)
        self.app.add_url_rule("/reconf", "reconf", self.reconf)
        self.request = request
        self.thread = threading.Thread(
            target=self.app.run,
            kwargs={"host": "0.0.0.0", "port": options.apiport},  # nosec
            daemon=True,
        )

    def start(self):
        self.thread.start()

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
