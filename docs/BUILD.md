# How to build a gamutRF system

## Prerequisites

- 1 x Raspberry Pi4 running Raspberry Pi OS (64-bit), A port of Debian Bullseye with the Raspberry Pi Desktop (Orchestrator)
- 1 x Raspberry Pi4 running Ubuntu 22.04.1 LTS Server (64-bit)
- 1 x [PoE Switch](https://www.amazon.com/gp/product/B087QQ46K4)
- 1 x [GPS module](https://www.adafruit.com/product/746)
- 1 x [7" Touchscreen](https://www.amazon.com/dp/B09X2N9C5V)
- 1 x [Triple Axis Compass](https://www.amazon.com/gp/product/B07PP67N9Q)
- 2 x [Ettus B200-mini](https://www.ettus.com/all-products/usrp-b200mini/)
- 1 x [USB 3.1 Flash Drive](https://www.amazon.com/dp/B07857Y17V)
- 1 x [Antenna Splitter](https://www.amazon.com/gp/product/B07STYNB6V)
- 1 x [Directional Antenna 850-6500Mhz](https://www.wa5vjb.com/products1.html)

## Installation overview

These instructions are for 2 different machines:
 - One that is the orchestrator which will do signal scanning, serve up GPS and heading, send out requests to workers to collect I/Q or RSSI sample for a given frequency, and run the [BirdsEye](https://github.com/IQTLabs/BirdsEye) interface.
 - One that is the worker which will do collection as instructed via API from the orchestrator.  There can be `n` number of workers assigned to a given orchestrator.

### Orchestrator

1. Install Raspberry Pi4 to 7" Touchscreen

2. Install GPS module to the 7" Touchscreen 40-pin header
```
PPS -> PIN12 (GPIO18)
VIN -> PIN17 (3.3V)
GND -> PIN25 (GND)
RX  -> PIN8  (TXD)
TX  -> PIN10 (RXD)
```

3. Install Triple Axis Compass to the 7" Touchscreen 40-pin header
```
VCC -> PIN1  (3.3V)
GND -> PIN6  (GND)
SCL -> PIN5  (SCL)
SDA -> PIN3  (SDA)
```

4. Plug in an Ettus B200 mini into a USB3 port on the Pi4.

5. Install Raspberry Pi OS (64-bit), a port of Debian Bullseye with the Raspberry Pi Desktop to the micro SD card.

6. Install dependencies:
```
sudo apt-get update
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
sudo apt install -y git libjpeg-dev python3 python3-pip python3-tk uhd-host gpsd gpsd-clients chrony pps-tools onboard at-spi2-core tmux
sudo /usr/lib/uhd/utils/uhd_images_downloader.py -t "b2|usb"
git clone https://github.com/IQTLabs/gamutRF
git clone https://github.com/IQTLabs/BirdsEye
sudo su -
mkdir -p /flash/gamutrf
echo "dtoverlay=vc4-kms-dsi-7inch" >> /boot/config.txt
echo "dtoverlay=vc4-kms-v3d" >> /boot/config.txt
echo "hdmi_cvt=1024 600 60 3 0 0 0" >> /boot/config.txt
echo "hdmi_group=2" >> /boot/config.txt
echo "hdmi_mode=87" >> /boot/config.txt
echo "hdmi_drive=2" >> /boot/config.txt
```

7. Set options in raspi-config
```
sudo raspi-config
-> Interface Options
  -> Serial Port
    -> Disable serial login
    -> Enable serial hardware
  -> I2C
    -> Enable 
```
Reboot

8. Install GamutRF and BirdsEye
```
cd gamutRF && docker compose -f orchestrator.yml pull && cd ..
cd BirdsEye && pip3 install -r requirements.txt && cd ..
```

9. Enable GPSD to listen on the external interface

Change `/lib/systemd/system/gpsd.socket` to match the following:
```
[Unit]
Description=GPS (Global Positioning System) Daemon Sockets

[Socket]
ListenStream=/run/gpsd.sock
#ListenStream=[::1]:2947
#ListenStream=127.0.0.1:2947
# To allow gpsd remote access, start gpsd with the -G option and
# uncomment the next two lines:
ListenStream=[::]:2947
ListenStream=0.0.0.0:2947
SocketMode=0600
#BindIPv6Only=yes

[Install]
WantedBy=sockets.target
```

10. Add BirdsEye systemd service

Create `/etc/systemd/system/birdseye.service` to contain the following:
```
[Unit]
Description=BirdsEye

[Service]
Type=simple
Restart=always
User=pi
Group=pi
WorkingDirectory=/home/pi/BirdsEye
Environment=DISPLAY=:0
ExecStart=/usr/bin/python3 sigscan.py

[Install]
WantedBy=multi-user.target
```
 
11. Add GPSD systemd service

Create `/etc/systemd/system/gpsd.service` to contain the following:
```
[Unit]
Description=GPSD

[Service]
Type=simple
Restart=always
ExecStart=/usr/sbin/gpsd -G -N -n /dev/serial0 -F /var/run/gpsd.sock

[Install]
WantedBy=multi-user.target
```

12. Add to the end of `/etc/chrony/chrony.conf`:
```
# SHM0 from gpsd is the NMEA data at 9600, so is not very accurate
refclock SHM 0  delay 0.5 refid NMEA
refclock PPS /dev/pps0 refid PPS

# SHM1 from gpsd (if present) is from the kernel PPS_LDISC
# module.  It includes PPS and will be accurate to a few ns
#refclock SHM 1 offset 0.0 delay 0.1 refid NMEA+
#
allow 192.168.111.0/24
```

13. Add to the following sections in `/etc/systemd/system/chronyd.service`:
```
[Unit]
StartLimitIntervalSec=30
StartLimitBurst=5

[Service]
Restart=on-failure
```

14. Enable the PPS GPIO:
```
sudo su -
echo 'pps-gpio' >> /etc/modules
```

15. Enable services
```
sudo systemctl enable birdseye.service
sudo systemctl enable gpsd.service
sudo systemctl enable gpsd.socket
sudo systemctl daemon-reload
```

16. Set static IP address for wired connection (plug ethernet into the PoE switch - non-PoE port)
```
sudo su -
echo 'interface eth0' >> /etc/dhcpcd.conf
echo 'static ip_address=192.168.111.10/24' >> /etc/dhcpcd.conf
```

17. Disable bluetooth and wifi (note you'll want to be wired into the switch and have an IP on the `192.168.111.0/24` subnet to maintain remote access to the Orchestrator after you do this)
```
sudo rfkill block bluetooth
sudo rfkill block wlan
```

18. Set any [keyboard configuration changes](https://docs.sunfounder.com/projects/ts-7c/en/latest/settings_for_raspberry_pi.html#install-virtual-keyboard-on-raspberry-pi)
```
Raspberry Pi Icon -> Preferences -> Onboard Settings
```

19. Reboot
```
sudo reboot
```

20. Start the orchestrator containers
```
cd gamutRF
UHD_IMAGES_DIR=/usr/share/uhd/images uhd_find_devices && VOL_PREFIX=/flash/gamutrf/ FREQ_START=70e6 FREQ_END=6e9 docker compose -f orchestrator.yml up -d
```
(optionally add `-f monitoring.yml` if you want additional monitoring containers)

Additionally, if you want to use the workers as recorders you'll want to update `orchestrator.yml` (before running the `docker compose` command above) under the gamutRF directory to include it. Multiple workers can be assigned to be recorders. Here's an exmaple with two:
```
  sigfinder:
    restart: always
    image: iqtlabs/gamutrf:latest
    networks:
      - gamutrf
    ports:
      - '80:80'
      - '9002:9000'
    volumes:
      - '${VOL_PREFIX}:/logs'
    command:
      - gamutrf-sigfinder
      - --scanners=sigfinder:8001
      - --log=/logs/scan.csv
      - --fftlog=/logs/fft.csv
      - --fftgraph=/logs/fft.png
      - --width=10
      - --prominence=2
      - --threshold=-25
      - '--freq-start=${FREQ_START}'
      - '--freq-end=${FREQ_END}'
      - --recorder=http://192.168.111.11:8000
      - --recorder=http://192.168.111.12:8000
```

### Worker

1. Plug in an Ettus B200 mini into a USB3 port on the Pi4.

2. Plug in the USB3.1 Flash drive into a USB3 port on the Pi4.

3. Install Ubuntu 22.04.1 LTS Server (64-bit) to the micro SD card (NOTE: Raspbian should also work, but has not been tested).

4. Install dependencies and configuration.
```
sudo apt-get update
sudo apt-get -y upgrade
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
sudo apt install -y git python3 python3-pip uhd-host
sudo /usr/lib/uhd/utils/uhd_images_downloader.py -t "b2|usb"
git clone https://github.com/IQTLabs/gamutRF
sudo su -
echo "dtoverlay=vc4-kms-v3d-pi4" >> /boot/firmware/config.txt
```

5. Setup Flash drive

Verify the device name using `lsblk -f` and create an ext4 filesystem on it:
```
sudo mkfs -t ext4 /dev/sda1
```

Copy the UUID of the device from `lsblk -f` (note it will have changed after running `mkfs`). Add the following (replacing the UUID value with your own) to `/etc/fstab`:
```
UUID=a04e77e2-772e-45b0-8590-bfb0741d855d /flash ext4 defaults,auto,users,rw,nofail 0 0
```

6. Set static IP address for wired connection (plug ethernet into the PoE switch - PoE port). Add the following to the end of `/etc/netplan/50-cloud-init.yaml`: 
```
    ethernets:
        eth0:
            addresses:
                - 192.168.111.11/24
```

7. Reboot
```
sudo reboot
```

8. Install GamutRF
```
cd gamutRF && docker compose -f worker.yml pull && cd ..
sudo mkdir -p /flash/gamutrf 
```

9. Choose what type of worker you want:

Each worker can be run in either `recorder` mode or `RSSI` mode.

If run in `recorder` mode (the default) no changes on the worker are needed, but the recorder needs to be added to the orchestrator as described above. In `recorder` mode the worker will capture full I/Q samples in `s16` format, and write it out to `/flash/gamutrf` as `.zst` compressed files.

If run in `RSSI` mode the `worker.yml` file under the gamutrf directory needs to be updated to include the following options:
```
  gamutrf-api:
    restart: always
    image: iqtlabs/gamutrf:latest
    networks:
      - gamutrf
    ports:
      - '8000:8000'
    cap_add:
      - SYS_NICE
      - SYS_RAWIO
    privileged: true
    devices:
      - /dev/bus/usb:/dev/bus/usb
    volumes:
      - '${VOL_PREFIX}:/data'
    environment:
      - 'WORKER_NAME=${WORKER_NAME}'
      - 'ORCHESTRATOR=${ORCHESTRATOR}'
      - 'CALIBRATION=${CALIBRATION}'
      - 'ANTENNA=${ANTENNA}'
    command:
      - nice
      - '-n'
      - '-19'
      - gamutrf-api
      - --no-agc
      - --rxb=62914560
      - '--gain=${GAIN}'
      - --rssi
      - --rssi_threshold=-110
      - --rssi_throttle=10
```
RSSI mode will only record signal strength in the form of float.

10. Start GamutRF
```
cd gamutRF
UHD_IMAGES_DIR=/usr/share/uhd/images uhd_find_devices && VOL_PREFIX=/flash/ ORCHESTRATOR=192.168.111.10 WORKER_NAME=worker1 ANTENNA=directional docker compose -f worker.yml up -d
```

## Initiating an API request

If running the orchestrator as a scanner with recorder workers, it will automatically make requests via the API for you based on signals it detects and the workers will start capturing I/Q. However if you want to control what gets recorded or are using workers in RSSI mode you'll need to make manual requests. Browse to the IP of the orchestrator via your browser and make a request with the specifications you desire.
