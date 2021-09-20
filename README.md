# gamutRF

gamutRF

# Quick Start

```
docker build -t gamut-rf:latest .
docker run --cap-add SYS_NICE --cap-add SYS_RAWIO --device /dev/bus/usb:/dev/bus/usb --privileged -v "$PWD":/logs -t gamut-rf:latest
```
