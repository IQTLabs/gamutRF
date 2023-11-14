# Troubleshooting

Instructions for troubleshooting common errors on GamutRF. If your error is not covered here please create an issue so the team can address it and update the documentation.

## Common Errors

### SoapySDR errors allocating buffers

Run ```echo 0 > /sys/module/usbcore/parameters/usbfs_memory_mb``` as root before starting the scanner(s).

### Containers won't start using Ettus SDRs

#### ```[ERROR] [USB] USB open failed: insufficient permissions```

Ettus SDRs download firmware and switch USB identities when first powered up. Restart the affected container to work around this (if run with docker compose, restart will happen automatically).

#### ```[ERROR] [UHD] An unexpected exception was caught in a task loop.The task loop will now exit, things may not work.boost: mutex lock failed in pthread_mutex_lock: Invalid argument```

UHD driver arguments ```num_recv_frames``` or ```recv_frame_size``` may be too high. The defaults are defined as ETTUS_ARGS in [utils.py](gamutrf/utils.py). Try reducing one or both via ```--sdrargs```. For example, ```--sdrargs num_recv_frames=64,recv_frame_size=8200,type=b200```.

#### ```[ERROR] [UHD] EnvironmentError: IOError: usb rx6 transfer status: LIBUSB_TRANSFER_OVERFLOW```

Stop containers, and reset the Ettus as follows:

```
$ /usr/lib/uhd/utils/b2xx_fx3_utils -D
$ /usr/lib/uhd/utils/b2xx_fx3_utils -U
$ /usr/lib/uhd/utils/b2xx_fx3_utils -S
```

### Scanner with Ettus SDR shows implausible low power at approx 100MHz intervals

Ettus radios periodically need extra time to produce good data when being retuned rapidly by the scanner. Increasing the value of ```--db_clamp_floor``` will cause the scanner to discard windows after retuning (effectively waiting for the retune command to be executed and produce good data before proceeding).

### "O"s or warnings about overflows in SDR containers

* Ensure your hardware can support the I/Q sample rate you have configured (gamutRF has been tested on Pi4 at 20Msps, which is the default recording rate). Also ensure your recording medium (e.g. flash drive, USB hard disk) is not timing out or blocking.
* If using a Pi4, make sure you are using active cooling and an adequate power supply (no CPU throttling), and you are using a "blue" USB3 port.