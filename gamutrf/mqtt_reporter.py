import gpsd
import json
import logging
import math
import paho.mqtt.client as mqtt
import smbus2
import socket


class MQTTReporter:

    def __init__(self, name, mqtt_server=None, gps_server=None, compass=False):
        self.name = name
        self.mqtt_server = mqtt_server
        self.compass = compass
        self.gps_server = gps_server
        self.mqttc = None
        self.bearing = 'no bearing'

    def connect(self):
        logging.info(f'connecting to {self.mqtt_server}')
        self.mqttc = mqtt.Client()
        self.mqttc.connect(self.mqtt_server)
        self.mqttc.loop_start()
        if self.gps_server:
            gpsd.connect(host=self.gps_server, port=2947)

    def get_bearing(self):
        bus = smbus2.SMBus(1)
        address = 0x0d
        self.write_byte(11, 0b00000001)
        self.write_byte(10, 0b00100000)
        self.write_byte(9, 0xD)
        scale = 0.92
        x_offset = -10
        y_offset = 10
        x_out = (self.read_word_2c(0)- x_offset+2) * scale  # calculating x,y,z coordinates
        y_out = (self.read_word_2c(2)- y_offset+2)* scale
        z_out = self.read_word_2c(4) * scale
        self.bearing = math.atan2(y_out, x_out)+.48  # 0.48 is correction value
        if(self.bearing < 0):
            self.bearing += 2* math.pi

    def read_byte(self, adr): # communicate with compass
        return bus.read_byte_data(address, adr)

    def read_word(self, adr):
        low = bus.read_byte_data(address, adr)
        high = bus.read_byte_data(address, adr+1)
        val = (high<< 8) + low
        return val

    def read_word_2c(self, adr):
        val = self.read_word(adr)
        if (val>= 0x8000):
            return -((65535 - val)+1)
        else:
            return val

    def write_byte(self, adr, value):
        bus.write_byte_data(address, adr, value)

    def add_gps(self, publish_args):
        if not self.gps_server:
            return publish_args
        publish_args.update({
            'position': [0, 0],
            'altitude': None,
            'gps_time': None,
            'map_url': None,
            'bearing': self.bearing,
            'gps': 'no fix'})
        try:
            if self.compass:
                self.get_bearing()
            packet = gpsd.get_current()
            publish_args.update({
                'position': packet.position(),
                'altitude': packet.altitude(),
                'gps_time': packet.get_time().timestamp(),
                'map_url': packet.map_url(),
                'bearing': self.bearing,
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
