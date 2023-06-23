import time
import pytest
from falcon import testing

from gamutrf import api


class FakeArgs:
    def __init__(self):
        self.name = "test"
        self.mqtt_server = ""
        self.qsize = 2
        self.sdr = "/dev/null"
        self.sdrargs = ""
        self.path = ""
        self.gain = -40
        self.agc = False
        self.rxb = 1e6
        self.freq_excluded = ""
        self.sigmf = False
        self.antenna = "0"
        self.rssi_throttle = 1
        self.rssi_threshold = -100
        self.rssi_external = False
        self.mean_window = 100
        self.rotate_secs = 3600


@pytest.fixture(scope="module")
def client():
    app = api.API(FakeArgs())
    return testing.TestClient(app.app)


def test_routes(client):
    result = client.simulate_get("/v1")
    assert result.status_code == 200
    result = client.simulate_get("/v1/info")
    assert result.status_code == 200
    result = client.simulate_get("/v1/record/100000000/20000000/20000000")
    assert result.status_code == 200


def test_report_rssi():
    app = api.API(FakeArgs())
    app.report_rssi({"center_freq": 1e6}, -35, time.time())


def test_serve_recording():
    app = api.API(FakeArgs())
    app.q.put({"center_freq": 1e6, "sample_count": 1e6})
    app.serve_recording(app.record)


def test_serve_rssi():
    app = api.API(FakeArgs())
    app.q.put({"center_freq": 1e6, "sample_count": 1e6, "sample_rate": 1e6})
    app.serve_rssi()


def test_argument_parse():
    api.argument_parser()
