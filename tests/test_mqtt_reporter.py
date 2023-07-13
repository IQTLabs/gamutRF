#!/usr/bin/python3
import pytest
import socket
import tempfile
import unittest

from gamutrf.mqtt_reporter import MQTTReporter

MQTT_PORT=1883

class FakeBroker:
    def __init__(self, port):
        # Bind to "localhost" for maximum performance, as described in:
        # http://docs.python.org/howto/sockets.html#ipc
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(30)
        sock.bind(("localhost", port))
        sock.listen()

        self._sock = sock
        self._conn = None

    def start(self):
        if self._sock is None:
            raise ValueError('Socket is not open')

        (conn, address) = self._sock.accept()
        conn.settimeout(10)
        self._conn = conn

    def finish(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def receive_packet(self, num_bytes):
        if self._conn is None:
            raise ValueError('Connection is not open')

        packet_in = self._conn.recv(num_bytes)
        return packet_in

    def send_packet(self, packet_out):
        if self._conn is None:
            raise ValueError('Connection is not open')

        count = self._conn.send(packet_out)
        return count


@pytest.fixture(scope="class")
def fake_broker():
    # print('Setup broker')
    broker = FakeBroker(MQTT_PORT)

    yield broker

    # print('Teardown broker')
    broker.finish()


def test_null_mqtt_reporter(fake_broker):
    with tempfile.TemporaryDirectory() as tempdir:
        mqtt_reporter = MQTTReporter("myname")
        mqtt_reporter.publish("/somewhere", {"doesnot": "matter"})
        mqtt_reporter = MQTTReporter("myname", mqtt_server="localhost")
        mqtt_reporter.publish("/somewhere", {"doesnot": "matter"})
        mqtt_reporter.log(tempdir, "test", 1, {"test": "data"})
        mqtt_reporter.log("/no/such/path", "test", 1, {"test": "data"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
