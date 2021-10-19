# gamutRF

An SDR orchestrated scanner.

# Quick Start

Build and run the collection and signal finding containers (note: an Ettus b2XX will need to be attached over USB):

```
docker-compose build
docker-compose up
```

Finally build and run the container for generating the graph video (MP4) from the CSV file:
```
docker build -t gamut-graph:latest -f Dockerfile.graph .
docker run -it --rm -v "$PWD":/data gamut-graph:latest /data/scan.csv
```

# FAQ

If you see the following error:
```
[ERROR] [USB] USB open failed: insufficient permissions.
See the application notes for your device.
```
Exit out of the container (`ctrl-c`, or `docker rm -f gamutrf`) and try running the container again.
