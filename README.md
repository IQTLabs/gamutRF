# gamutRF

An SDR orchestrated scanner.

# Prerequisites

- Linux (Ubuntu 20.04+)
- Python3
- pip3
- git
- gpsd and a GPS module on the orchestrator
- [Docker](https://docs.docker.com/engine/install/ubuntu/)
- docker-compose (`pip3 install docker-compose`)
- UHD (if using Ettus):
```
sudo apt-get install uhd-host
sudo /usr/lib/uhd/utils/uhd_images_downloader.py -t "b2|usb"
```

# Quick Start

This project is designed to run across several machines, one `orchestrator` and `n` number of `workers`.

Clone the project on the `orchestrator`:
```
git clone https://github.com/IQTLabs/gamutRF.git
cd gamutRF
```

Change the example `recorder` line under sigfinder in `docker-compose-orchestrator.yml` to match the IP or name of the `worker`. Add additional `recorder` lines for multiple `workers`. Here's an example with two workers:
```
  sigfinder:
    restart: always
    image: iqtlabs/gamutrf-sigfinder:latest
    build:
      context: .
      dockerfile: Dockerfile.sigfinder
    networks:
      - gamutrf
    ports:
      - '9002:9000'
    volumes:
      - '${VOL_PREFIX}:/logs'
    command:
      - --logaddr=sigfinder
      - --log=/logs/scan.csv
      - --recorder=http://192.168.111.11:8000
      - --recorder=http://192.168.111.12:8000
      - '--freq-start=${FREQ_START}'
      - '--freq-end=${FREQ_END}'
```

Build and run the collection and signal finding containers (note: an Ettus b2XX will need to be attached over USB) on the `orchestrator` (optionally override the `VOL_PREFIX` for where data will be stored and the `FREQ_START` and `FREQ_END` to change the spectrum range that gets scanned:
```
VOL_PREFIX=/flash/gamutrf/ FREQ_START=70e6 FREQ_END=6e9 docker-compose -f docker-compose-orchestrator.yml up -d
```

Finally build and run the collection containers on each of the workers:
```
git clone https://github.com/IQTLabs/gamutRF.git
cd gamutRF
VOL_PREFIX=/flash/ ORCHESTRATOR=192.168.111.10 docker-compose -f docker-compose-worker.yml up -d
```

# FAQ

If you see the following error:
```
[ERROR] [USB] USB open failed: insufficient permissions.
See the application notes for your device.
```
Exit out of the container (`ctrl-c`, or `docker rm -f gamutrf`) and try running the container again.
