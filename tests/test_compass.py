import sys

import fake_rpi

sys.modules['smbus2'] = fake_rpi.smbus
from gamutrf import compass


def test_compass_heading():
    heading = compass.Heading()
    heading.get_heading()
