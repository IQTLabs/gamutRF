#!/usr/bin/python3

import tempfile
import unittest

from gamutrf.mqtt_reporter import MQTTReporter


class MQTTReporterTestCase(unittest.TestCase):
    def test_null_mqtt_reporter(self):
        with tempfile.TemporaryDirectory() as tempdir:
            mqtt_reporter = MQTTReporter("myname")
            mqtt_reporter.publish("/somewhere", {"doesnot": "matter"})
            mqtt_reporter = MQTTReporter("myname", mqtt_server="localhost")
            mqtt_reporter.publish("/somewhere", {"doesnot": "matter"})
            mqtt_reporter.log(tempdir, "test", 1, {"test": "data"})
            mqtt_reporter.log("/no/such/path", "test", 1, {"test": "data"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
