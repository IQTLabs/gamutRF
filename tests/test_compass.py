from gamutrf import compass
import sys

import fake_rpi

sys.modules['smbus2'] = fake_rpi.smbus


def test_compass_bearing():
    bearing = compass.Bearing()
    bearing.get_bearing()
