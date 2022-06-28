import logging
import math

import bjoern
import falcon
import smbus2
from falcon_cors import CORS


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class Heading:

    def get_heading(self, calibration=0):
        self.bus = smbus2.SMBus(1)
        self.address = 0x0d
        self.heading_reading = 'no heading'
        self.write_byte(11, 0b00000001)
        self.write_byte(10, 0b00100000)
        self.write_byte(9, 0xD)
        scale = 0.92
        x_offset = -10
        y_offset = 10
        x_out = (self.read_word_2c(0) - x_offset+2) * \
            scale  # calculating x,y,z coordinates
        y_out = (self.read_word_2c(2) - y_offset+2) * scale
        z_out = self.read_word_2c(4) * scale
        self.heading_reading = math.atan2(
            y_out, x_out)+.48  # 0.48 is correction value
        if(self.heading_reading < 0):
            self.heading_reading += 2 * math.pi
        # convert to degrees
        self.heading_reading = (self.heading_reading * 180) / math.pi
        self.heading_reading = (self.heading_reading + calibration) % 360

    def read_byte(self, adr):  # communicate with compass
        return self.bus.read_byte_data(self.address, adr)

    def read_word(self, adr):
        low = self.bus.read_byte_data(self.address, adr)
        high = self.bus.read_byte_data(self.address, adr+1)
        val = (high << 8) + low
        return val

    def read_word_2c(self, adr):
        val = self.read_word(adr)
        if (val >= 0x8000):
            return -((65535 - val)+1)
        else:
            return val

    def write_byte(self, adr, value):
        self.bus.write_byte_data(self.address, adr, value)

    def on_get(self, _req, resp, calibration):
        self.get_heading(calibration=int(calibration))
        resp.text = str(self.heading_reading)
        resp.content_type = falcon.MEDIA_TEXT
        resp.status = falcon.HTTP_200


class CompassAPI:

    def __init__(self):
        cors = CORS(allow_all_origins=True)
        self.app = falcon.App(middleware=[cors.middleware])
        self.main()

    @staticmethod
    def paths():
        return ['/{calibration}']

    @staticmethod
    def version():
        return '/v1'

    def routes(self):
        p = self.paths()
        heading = Heading()
        funcs = [heading]
        return dict(zip(p, funcs))

    def main(self):
        logging.info('adding API routes')
        r = self.routes()
        for route in r:
            self.app.add_route(self.version()+route, r[route])

        logging.info('starting API thread')
        bjoern.run(self.app, '0.0.0.0', 8000)


if __name__ == "__main__":
    CompassAPI()
