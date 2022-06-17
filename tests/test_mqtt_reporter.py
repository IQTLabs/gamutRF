#!/usr/bin/python3
import unittest

from gamutrf.mqtt_reporter import MQTTReporter


class MQTTReporterTestCase(unittest.TestCase):

    def test_null_mqtt_reporter(self):
        mqtt_reporter = MQTTReporter('myname')
        mqtt_reporter.publish('/somewhere', {'doesnot': 'matter'})
        mqtt_reporter = MQTTReporter('myname', mqtt_server='localhost')
        mqtt_reporter.publish('/somewhere', {'doesnot': 'matter'})


if __name__ == '__main__':
    unittest.main()
