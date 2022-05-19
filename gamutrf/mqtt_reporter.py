
import json
import logging
import socket
import gpsd
import paho.mqtt.client as mqtt


class MQTTReporter:

    def __init__(self, name, mqtt_server=None, gps_server=None):
        self.name = name
        self.mqtt_server = mqtt_server
        self.gps_server = gps_server
        self.mqttc = None

    def connect(self):
        logging.info(f'connecting to {self.mqtt_server}')
        self.mqttc = mqtt.Client()
        self.mqttc.connect(self.mqtt_server)
        self.mqttc.loop_start()
        if self.gps_server:
            gpsd.connect(host=self.gps_server, port=2947)

    def add_gps(self, publish_args):
        if not self.gps_server:
            return
        publish_args.update({
            'position': [0, 0],
            'altitude': None,
            'gps_time': None,
            'map_url': None,
            'gps': 'no fix'})
        try:
            packet = gpsd.get_current()
            publish_args.update({
                'position': packet.position(),
                'altitude': packet.altitude(),
                'gps_time': packet.get_time().timestamp(),
                'map_url': packet.map_url(),
                'gps': 'fix'})
        except (gpsd.NoFixError, AttributeError) as err:
            logging.error('could not update with GPS: %s', err)
        return publish_args

    def publish(self, publish_path, publish_args):
        if not self.mqtt_server:
            return
        try:
            if self.mqttc is None:
                self.connect()
            publish_args = self.add_gps(publish_args)
            publish_args['name'] = self.name
            self.mqttc.publish(publish_path, json.dumps(publish_args))
        except (socket.gaierror, ConnectionRefusedError, mqtt.WebsocketConnectionError, ValueError) as err:
            logging.error(f'failed to publish to MQTT {self.mqtt_server}: {err}')
