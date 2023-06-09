# gamutRF

An SDR orchestrated scanner and collector.

gamutRF is a system enabling a compact network of one or more modest machines (such as Pi4s), each with their own USB SDR (such as an Ettus
B200mini or a BladeRF XA9), to operate collectively as a configurable wideband scanner and I/Q sample recorder.

A gamutRF system comprises an "orchestrator" machine which typically runs at least one scanner service (typically scanning 0.1GHz to 6GHz in 30s) and the sigfinder service, which then
can command potentially many gamutRF "workers" to make I/Q sample recordings for later analysis. sigfinder can command multiple scanners which can be on the same machine or be connected
over a network.

gamutRF provides tools to work with I/Q sample recordings, and to also record GPS location/compass metadata for the system itself. gamutRF typically runs on networks of Raspberry Pi4s, but can also run on x86 machines, and is based on gnuradio.

See also [instructions on how to build a gamutRF system](BUILD.md).

## Scanner theory of operation

gamutRF's scanner container connects to a local SDR and sweeps over a configured frequency range or ranges collecting samples. The samples are sent an FFT block, and then to a [streaming retuning gnuradio block](https://github.com/iqtlabs/gr-iqtlabs), which aggregates and then serves the frequency-annotated FFT points as JSON objects over ZMQ to the "sigfinder" container (see below) and also commands the source SDR block to retune to a new frequency in the range when enough FFT points have been accumulated.

The sigfinder container consumes these FFT points over ZMQ (potentially from many scanners), does some noise processing (correcting FFT points to be in frequency order, computing mean power over 10kHz, and then a rolling mean over 1MHz) and then submits them to [scipy.signals.find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html).

If workers have been provisioned, the sigfinder will then command the workers to make an approximately 10 second I/Q recording at approximately 20Msps of each signal. Each signal peak is assigned a 20MHz bin, which means that if a signal is repeatedly detected with some frequency variation, the assigned recording bin will be constant, and if multiple signals are detected within 20MHz they can be collected simultaneously. A worker by default records at a higher sample rate than the bin size, so that 20MHz signal margins can be recorded.

As there will almost certainly be more signals than workers available, sigfinder will prioritize signals that it least often observed over a configurable number of scanner cycles. It is possible to configure this to `off` so that the recording choice will be random. It is also possible to configure the workers to tell the sigfinder to exclude that worker from certain frequency ranges (if for example the worker SDR cannot handle some part of the frequency spectrum scanned).

## Operating gamutRF

See the [build doc](BUILD.md)

### SDR/scanner/sigfinder command line options

While there are other options, these options primarily influence gamutRF's scanner functionality.

#### scanner

| Option | Description |
| -- | -- |
| --freq-start and --freq-end | Start and end of frequency range to scan in Hz |
| --tuning_ranges | Overrides --freq-start and --freq-end if present. A comma separated list of ranges in Hz to scan, for example ```2.2e9-2.6e9,5.1e9-5.9e9``` |
| --igain | SDR input gain in dB |
| --samp-rate | Number of samples/sec |
| --tuneoverlap | For each retune interval, advance center frequency by N * --samp_rate |
| --sweep-sec | Attempt to sweep frequency range within N seconds (effectively, best guess for --tune-step-fft) |
| --tune-step-fft | Overrides --sweep-sec if present. Retune to next interval every N FFT points |
| --skip-tune-step | Discard N FFT points after retuning |
| --nfft | Number of FFT points |
| --write_samples | If > 0, write N samples to disk after retuning, in --sample_dir as a zstd compressed file |
| --description | Optionally provide text description along with streaming scanner updates, to the sigfinder |

##### sigfinder

| Option | Description |
| -- | -- |
| --width | Minimum width of a peak to be detected in 0.01MHz increments (passed to scipy find_peaks()) |
| --prominence | Minimum prominence of a peak to be detected (passed to scipy find_peaks()) |
| --threshold | Minimum threshold in dB of a peak to be detected (passed to scipy find_peaks()) |
| --bin_width | Bin width in MHz |
| --history | Number of scanner cycles over which to prioritize recording of least often observed peaks |
| --record_bw_msps | Number of samples per second in units of 1024^2 (generally larger than bin size to record signal margins) |
| --record_secs | Length of time to record I/Q samples in seconds |
| --fftlog | Log raw output of CSV to this file, which will be rotated every --rotatesecs |
| --fftgraph | Graph the most recent FFT signal and peaks to this PNG file (will keep the last --nfftgraph versions) |
| --scanners | Comma separated list of scanner ZMQ endpoints to use, for example --scanners=127.0.0.1:8001,127.0.0.1:9002 |

### Running multiple radios on the same machine

gamutRF supports the use of multiple radios on the same machine, whether for scanning or recording. When using multiple Ettus SDRs, assign a specific radio to a specific container by specifying the ```serial``` UHD driver argument. To list all connected Ettus radios, run ```uhd_find_devices```. Then add the ```--sdrargs``` argument to the specific container. For example, ```--sdrargs num_recv_frames=960,recv_frame_size=16360,type=b200,serial=12345678```.

### Manually initiating worker actions

The orchestrator has a web interface on port 80. You can use this to command a worker to start an I/Q sample recording or start a RSSI stream.

## Working with worker I/Q recordings

Workers make recordings that are compressed with zstandard, and are typically in complex number, int16 format, and include the center frequency and sample rate that the recording was made with. gamutRF tools can generally work with such files directly, but other tools require the recordings to be converted (see below).

### Generating a spectrogram of a recording

gamutRF provides a tool to convert a recording or directory of recordings into a spectrogram. For example, to convert all I/Q recordings in /tmp:

```docker run -ti -v /tmp:/tmp iqtlabs/gamutrf gamutrf-specgram /tmp```

Use the ```--help``` option to change how the spectogram is generated (for example, to change the sample rate).

### Translating recordings to "gnuradio" format

Most SDR tools by convention take an uncompressed raw binary file as input, of [gnuradio type complex](https://blog.sdr.hu/grblocks/types.html). The user must explicitly specify to most SDR tools what sample rate the file was made at to correctly process it. gamutRF provides a tool that converts a gamutRF I/Q recording (which may be compressed) to an uncompressed binary file. For example:

```
docker run -v /tmp:/tmp -ti iqtlabs/gamutrf gamutrf-samples2raw /tmp/gamutrf_recording_ettus_directional_gain70_1234_100000000Hz_20971520sps.s16.zst
```

### Reviewing a recording interactively in gqrx

[gqrx](https://gqrx.dk/) is a multiplatform open source tool that allows some basic SDR operations like visualizing or audio demodulating an I/Q sample recording (see the [github releases page](https://github.com/gqrx-sdr/gqrx/releases), for a MacOS .dmg file). To use gqrx with a gamutRF recording, first translate the recording to gnuradio format (see above). Then open gqrx.

* Select ```Complex Sampled (I/Q) File```
* Set ```Input rate``` to be the same as the gamutRF sample rate (e.g. from the recording file name,
```gamutrf_recording_ettus_directional_gain70_1234_100000000Hz_20971520sps.raw```, set ```Input rate``` to 20971520, and also edit ```rate=``` in ```Device string``` to be 20971520)
* Set ``Bandwidth`` to 0
* Edit ```Device string``` to set the ```file=``` to be the path to the recording.
* Set ```Decimation``` to None.
* Finally select ```OK``` and then ```play``` from the gqrx interface to watch the recording play.

### Reducing recording sample rate

You may want to reduce the sample rate of a recording or re-center it with respect to frequency (e.g. to use another demodulator tool that doesn't support a high sample rate). gamutRF provides the ```freqxlator``` tool to do this.

* Translate your gamutRF recording to gnuradio format (see above).
* Use ```freqxlator``` to create a new recording at a lower sample rate, potentially with a different center frequency.

For example, to reduce a recording made with gamutRF's default sample rate to 1/10th the rate while adjusting the center frequency down by 1MHz, use:

```docker run -ti iqtlabs/gamutrf gamutrf-freqxlator --samp-rate 20971520 --center -1e6 --dec 10 gamutrf_recording_gain70_1234_100000000Hz_20971520sps.raw gamutrf_recording_gain70_1234_100000000Hz_2097152sps.raw```

### Demodulating AM/FM audio from a recording

gamutRF provides a tool to demodulate AM/FM audio from a recording as an example use case.

* Use the ```freqxlator``` tool to make a new recording at no more than 1Msps and has the frequency to be demodulated centered.
* Use the ```airspyfm``` tool to demodulate audio to a WAV file.

For example, to decode an FM recording which must be at the center frequency of a recording:

```docker run -v /tmp:/tmp -ti iqtlabs/gamutrf-airspyfm -m fm -t filesource -c filename=/tmp/gamutrf_recording_gain70_1234_100000000Hz_2097152sps.raw,raw,format=FLOAT,srate=2097152 -F /tmp/out.wav```

Run:

```docker run -ti iqtlabs/gamutrf-airspyfm -h```

To view other options.

## API access

gamutRF supports two separate APIs - for receiving scanner updates, and making scanner configuration changes.

* scanner update API: described in [zmqreceiver.py](gamutrf/zmqreceiver.py). Receives real time scanner updates and config.
* scanner config API: allows RESTful updates to any CLI argument. Eg, ```wget -O- "http://localhost:9001/reconf?freq_start=1e9&freq_end=2e9"``` causes the scanner to reset and scan 1e9 to 2e9Hz. Any config change causes the scanner's gnuradio flowgraph to restart.

## Scanner testing

Currently, the scanner ```gain``` and sigfinder ```threshold``` must be set manually for the general RF environment (e.g. noisy/many signals versus quiet/few signals).
To establish the correct values and to confirm the scanner is working, initiate a scan over the 2.2-2.6GHz range. As the 2.4GHz spectrum is very busy with legacy WiFi
and BlueTooth, the probability of seeing signals is high. If in an environment without BlueTooth or WiFi, an alternative is the FM broadcast band (88MHz to 108MHz).

To begin, commence scanning with just the scanner and sigfinder containers:

```
$ VOL_PREFIX=/tmp FREQ_START=2.2e9 FREQ_END=2.6e9 docker compose -f orchestrator.yml up gamutrf sigfinder
```

Watch for ```/tmp/fft.png``` to appear, which should contain strong signals similar to this example:

![2.4G example](fft24test.png)

If no or only small peaks appear which are not marked as peaks, increase ```gain``` (e.g., from 40 to 45) until peaks are detected.

If no peaks appear still, check antenna cabling, or choose a different scan range where signals are expected in your environment.

If peaks appear but are consistently not marked, decrease ```theshold``` (e.g. -25 to -35). If too many peaks are detected (noise detected as peaks), raise ```threshold.```

## Troubleshooting

#### Containers won't start using Ettus SDRs

##### ```[ERROR] [USB] USB open failed: insufficient permissions```

Ettus SDRs download firmware and switch USB identities when first powered up. Restart the affected container to work around this.

##### ```[ERROR] [UHD] An unexpected exception was caught in a task loop.The task loop will now exit, things may not work.boost: mutex lock failed in pthread_mutex_lock: Invalid argument```

UHD driver arguments ```num_recv_frames``` or ```recv_frame_size``` may be too high. The defaults are defined as ETTUS_ARGS in [utils.py](gamutrf/utils.py). Try reducing one or both via ```--sdrargs```. For example, ```--sdrargs num_recv_frames=64,recv_frame_size=8200,type=b200```.

#### "O"s or warnings about overflows in SDR containers

* Ensure your hardware can support the I/Q sample rate you have configured (gamutRF has been tested on Pi4 at 20Msps, which is the default recording rate). Also ensure your recording medium (e.g. flash drive, USB hard disk) is not timing out or blocking.
* If using a Pi4, make sure you are using active cooling and an adequate power supply (no CPU throttling), and you are using a "blue" USB3 port.
