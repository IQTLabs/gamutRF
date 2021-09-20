# gamutRF

An SDR orchestrated scanner.

# Quick Start

Build and run the collection container (note: an Ettus b2XX will need to be attached over USB):
```
docker build -t gamut-rf:latest .
docker run --cap-add SYS_NICE --cap-add SYS_RAWIO --device /dev/bus/usb:/dev/bus/usb -v "$PWD":/logs --name gamutrf -t gamut-rf:latest
```

When finished collecting, simply kill the container with `ctrl-c` or `docker rm -f gamutrf`.

Finally build and run the container for generating the graph video (MP4) from the CSV file:
```
docker build -t gamut-graph:latest -f Dockerfile.graph .
docker run -it --rm -v "$PWD":/data gamut-graph:latest /data/ettus_scan.csv
```

# FAQ

If you see the following error:
```
[ERROR] [USB] USB open failed: insufficient permissions.
See the application notes for your device.
```
Exit out of the container (`ctrl-c`, or `docker rm -f gamutrf`) and try running the container again.

