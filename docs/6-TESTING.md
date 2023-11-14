# GamutRF Testing

## Scanner testing

Currently, the scanner ```gain``` must be set manually for the general RF environment (e.g. noisy/many signals versus quiet/few signals).
To establish the correct values and to confirm the scanner is working, initiate a scan over the 2.3-2.6GHz range. As the 2.4GHz spectrum is very busy with legacy WiFi
and BlueTooth, the probability of seeing signals is high. If in an environment without BlueTooth or WiFi, an alternative is the FM broadcast band (88MHz to 108MHz).

To begin, commence scanning with just the scanner and waterfall containers:

```
VOL_PREFIX=/tmp FREQ_START=2.3e9 FREQ_END=2.6e9 docker compose -f orchestrator.yml up gamutrf waterfall
```

Browse the waterfall container's webserver port (by default, localhost:9003). It should contain strong signals similar to this example:

![2.4G example](imgs/fft24test.png)

If no or only small peaks appear which are not marked as peaks, increase ```gain``` (e.g., from 40 to 45) until peaks are detected.

If no peaks appear still, check antenna cabling, or choose a different scan range where signals are expected in your environment.