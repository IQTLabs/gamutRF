import json
import logging
import os
import socket
import time

import gpsd
import paho.mqtt.client as mqtt
from gamutrf.utils import http_get


class MQTTReporter:
    def __init__(
        self,
        name,
        mqtt_server=None,
        gps_server=None,
        compass=False,
        use_external_gps=False,
        use_external_heading=False,
        external_gps_server=None,
        external_gps_server_port=None,
    ):
        self.name = name
        self.mqtt_server = mqtt_server
        self.compass = compass
        self.gps_server = gps_server
        self.mqttc = None
        self.heading = "no heading"
        self.use_external_gps = use_external_gps
        self.use_external_heading = use_external_heading
        self.external_gps_server = external_gps_server
        self.external_gps_server_port = external_gps_server_port
        self.external_gps_msg = None
        self.gps_configured = True
        if not self.gps_server and not self.external_gps_server:
            logging.error("mqtt enabled, no gps_server or external_gps_server found")
            self.gps_configured = False
        if (
            self.external_gps_server
            and not self.use_external_gps
            and not self.gps_server
        ):
            logging.error(
                "mqtt enabled and only external_gps_server found, but no use_external_gps flag"
            )
            self.gps_configured = False

    @staticmethod
    def log(path, prefix, start_time, record_args):
        try:
            with open(
                os.path.join(path, f"mqtt-{prefix}-{start_time}.log"),
                "a+",
                encoding="utf-8",
            ) as f:
                f.write(f"{json.dumps(record_args)}\n")
        except FileNotFoundError as err:
            logging.error(f"could not write to mqtt rssi log: {err}")

    def connect(self):
        logging.info(f"connecting to {self.mqtt_server}")
        self.mqttc = mqtt.Client()
        self.mqttc.connect(self.mqtt_server)
        self.mqttc.loop_start()

    def get_heading(self):
        if self.use_external_heading:
            heading_result = http_get(
                f"http://{self.external_gps_server}:{self.external_gps_server_port}/heading"
            )
            if heading_result is None:
                logging.error("could not update external heading")
            else:
                self.heading = float(json.loads(heading_result.text)["heading"])
        else:
            heading_result = http_get(f"http://{self.gps_server}:8000/v1/heading")
            if heading_result is None:
                logging.error("could not update heading")
            else:
                self.heading = str(float(heading_result.text))

    def add_gps(self, publish_args):
        if not self.gps_configured:
            return publish_args
        publish_args.update(
            {
                "position": [0, 0],
                "altitude": None,
                "gps_time": None,
                "map_url": None,
                "heading": self.heading,
                "gps": "no fix",
            }
        )

        # Use external external GPS
        if self.use_external_gps:
            external_gps_msg = http_get(
                f"http://{self.external_gps_server}:{self.external_gps_server_port}/gps-data"
            )
            if external_gps_msg is None:
                logging.error("could not update with external GPS")
            else:
                self.external_gps_msg = json.loads(external_gps_msg.text)
                publish_args.update(
                    {
                        "position": (
                            self.external_gps_msg["latitude"],
                            self.external_gps_msg["longitude"],
                        ),
                        "altitude": self.external_gps_msg["altitude"],
                        "gps_time": self.external_gps_msg["time_usec"],
                        "map_url": None,
                        "heading": self.heading,
                        "gps": "fix",
                    }
                )

        # Use internal GPIO GPS
        else:
            try:
                if self.compass:
                    self.get_heading()
                if gpsd.gpsd_stream is None:
                    gpsd.connect(host=self.gps_server, port=2947)
                packet = gpsd.get_current()
                publish_args.update(
                    {
                        "position": packet.position(),
                        "altitude": packet.altitude(),
                        "gps_time": packet.get_time().timestamp(),
                        "map_url": packet.map_url(),
                        "heading": self.heading,
                        "gps": "fix",
                    }
                )
            except (BrokenPipeError, gpsd.NoFixError, AttributeError) as err:
                logging.error("could not update with GPS: %s", err)
        return publish_args

    def publish(self, publish_path, publish_args):
        if not self.mqtt_server:
            return
        try:
            if self.mqttc is None:
                self.connect()
            publish_args = self.add_gps(publish_args)
            publish_args["name"] = self.name
            self.mqttc.publish(publish_path, json.dumps(publish_args))
        except (
            socket.gaierror,
            ConnectionRefusedError,
            mqtt.WebsocketConnectionError,
            ValueError,
        ) as err:
            logging.error(f"failed to publish to MQTT {self.mqtt_server}: {err}")
