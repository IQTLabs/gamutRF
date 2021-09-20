# gamutRF

gamutRF

# Quick Start

```
docker build -t gamut-rf:latest .
docker run --cap-add SYS_NICE --cap-add SYS_RAWIO --device /dev/bus/usb:/dev/bus/usb -v "$PWD":/logs --name gamutrf -t gamut-rf:latest
```

# FAQ

If you see the following error:
```
[ERROR] [USB] USB open failed: insufficient permissions.
See the application notes for your device.
```
Exit out of the container (`ctrl-c`, or `docker rm -f gamutrf`) and try running the container again.

