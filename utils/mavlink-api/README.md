# Mavlink API Interface

## Setup and calibrate device

Use Mission Planner or QGroundControl to calibrate and configure device. Set the AHRS to the correct orientation for use.

## Set up device mount
Find the correct device serial port and attributes with the following:
```bash
lsusb
dmesg | grep tty
udevadm info -a /dev/ttyAMC0  #<--use correct serial port>
```

Test if you have the correct port:
```bash
python3 utils/mavlink_serial_test.py /dev/ttyACM9
```

Create a udev rule to map the hardware to a custom port
```bash
echo "KERNEL=="ttyAMC0", ATTRS{idVendor}=="3162", MODE:="0666", SYMLINK+="pixhawk_serial"" >> /etc/udev/rules.d/60-gamutrf.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Test again if it is working correctly:
```bash
python3 utils/mavlink_serial_test.py /dev/pixhawk_serial
```