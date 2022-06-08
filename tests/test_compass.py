import sys
import fake_rpi

sys.modules['smbus'] = fake_rpi.smbus

from gamutrf import compass

def test_compass_bearing():
    bearing = compass.Bearing()
    bearing.get_bearing()
