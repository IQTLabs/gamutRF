#!/usr/bin/python3
import glob
import json
import os
import tempfile
import time
import unittest

import docker
import numpy as np
import requests

from gamutrf.birdseye_rssi import BirdsEyeRSSI


class FakeArgs:

    def __init__(self):
        self.sdr = 'file:/dev/zero'
        self.threshold = -100
        self.mean_window = 4096
        self.rssi_threshold = -100
        self.gain = 10


class BirdseyeRSSITestCase(unittest.TestCase):

    def test_birdseye_smoke(self):
        tb = BirdsEyeRSSI(FakeArgs(), 1e3, 1e3)
        tb.start()
        time.sleep(1)
        tb.stop()
        tb.wait()

    def test_birdseye_endtoend_rssi(self):
        test_tag = 'iqtlabs/gamutrf-api:latest'
        with tempfile.TemporaryDirectory() as tempdir:
            testraw = os.path.join(tempdir, 'test.raw')
            gamutdir = os.path.join(tempdir, 'gamutrf')
            testdata = np.random.random(int(1e6)).astype(
                np.float32) + np.random.random(int(1e6)).astype(np.float32) * 1j
            with open(testraw, 'wb') as testrawfile:
                testdata.tofile(testrawfile)
            os.mkdir(gamutdir)
            client = docker.from_env()
            client.images.build(dockerfile='Dockerfile.api',
                                path='.', tag=test_tag)
            container = client.containers.run(
                test_tag,
                command=['--rssi_threshold=-100', '--rssi', '--sdr=file:/data/test.raw'],
                ports={'8000/tcp': 8000},
                volumes={tempdir: {'bind': '/data', 'mode': 'rw'}},
                detach=True)
            for _ in range(15):
                try:
                    response = requests.get(
                        'http://localhost:8000/v1/record/100000000/1000000/1000000')
                    self.assertEqual(200, response.status_code, response)
                    break
                except requests.exceptions.ConnectionError:
                    time.sleep(1)
            mqtt_logs = None
            for _ in range(15):
                mqtt_logs = glob.glob(os.path.join(gamutdir, 'mqtt-rssi-*log'))
                if mqtt_logs:
                    container.kill()
                    break
                time.sleep(1)
            self.assertTrue(mqtt_logs)
            with open(mqtt_logs[0], 'r') as log:
                for line in log.readlines():
                    line_json = json.loads(line)
                    self.assertGreater(-10, line_json['rssi'])


if __name__ == '__main__':
    unittest.main()
