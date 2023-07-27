import json
import logging
import os
import socket
import time

import gpsd
import httpx
import paho.mqtt.client as mqtt


class MQTTReporter:
    def __init__(self, name, mqtt_server=None, gps_server=None, compass=False, use_mavlink_gps=False):
        self.name = name
        self.mqtt_server = mqtt_server
        self.compass = compass
        self.gps_server = gps_server
        self.mqttc = None
        self.heading = "no heading"
        self.use_mavlink_gps = use_mavlink_gps
        self.mavlink_gps_topic = '/MAVLINK-GPS'
        self.mavlink_gps_msg = None
        self.mavlink_max_wait_time_s = 3

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
        if self.use_mavlink_gps:
            self.mqttc.subscribe(self.mavlink_gps_topic)
            self.mqttc.on_message = self.mavlink_gps_msg_callback
        self.mqttc.loop_start()    

    def get_heading(self):
        try:
            self.heading = str(
                float(httpx.get(f"http://{self.gps_server}:8000/v1/heading").text)
            )
        except Exception as err:
            logging.error("could not update heading: %s", err)

    def add_gps(self, publish_args):
        if not self.gps_server:
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
        try:
            #Use external MAVLINK GPS
            if self.use_mavlink_gps:
                self.mqttc.publish(self.mavlink_gps_topic, json.dumps({"msg":"GPS"}))
                start_time=time.time()
                while self.mavlink_gps_msg == None and time.time() - start_time < self.mavlink_max_wait_time_s:
                    time.sleep(0.001)
                if self.mavlink_gps_msg == None:
                    return publish_args
                else:
                    publish_args.update(
                        {
                            "position": self.mavlink_gps_msg["lat"]+","+self.mavlink_gps_msg["lon"],
                            "altitude": self.mavlink_gps_msg["alt"],
                            "gps_time": self.mavlink_gps_msg["time_usec"],
                            "map_url": None,
                            "heading": self.mavlink_gps_msg["hdg"],
                            "gps": "fix",
                        }
                    )
                self.heading=self.mavlink_gps_msg["hdg"]
                self.mavlink_gps_msg = None

            #Use internal GPIO GPS
            else:
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
    
    def mavlink_gps_msg_callback(self, client, userdata, msg):
        gps_msg=json.loads(msg)
        self.mavlink_gps_msg

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
