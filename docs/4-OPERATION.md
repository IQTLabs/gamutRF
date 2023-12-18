# Operating GamutRF

This section details the operation of a GamutRF system and how to configure the system (both orchestrator and worker) to achieve common use cases.

<ins>Table of Contents:<ins/>
- [CLI Options](#sdrscannersigfinder-command-line-options)
- [Common Use Cases](#common-operation-use-cases)
- [Running multiple radios on single maching](#running-multiple-radios-on-the-same-machine)
- [Manually initiating worker actions](#manually-initiating-worker-actions)
- [Utility Functions](#utility-functions)

## SDR/scanner/sigfinder command line options

While there are other options, these options primarily influence gamutRF's scanner functionality.

### scanner

| Option | Description |
| -- | -- |
| --freq-start and --freq-end | Start and end of frequency range to scan in Hz (if --freq-end=0, will run in "stare" mode, static scanning from --freq-start to --freq_start + (--tuneoverlap * --samp_rate))|
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

### sigfinder

| Option | Description |
| -- | -- |
| --bin_width | Bin width in MHz |
| --history | Number of scanner cycles over which to prioritize recording of least often observed peaks |
| --record_bw_msps | Number of samples per second in units of 1024^2 (generally larger than bin size to record signal margins) |
| --record_secs | Length of time to record I/Q samples in seconds |
| --fftlog | Log raw output of CSV to this file, which will be rotated every --rotatesecs |
| --scanners | Comma separated list of scanner ZMQ endpoints to use, for example --scanners=127.0.0.1:8001,127.0.0.1:9002 |

## Common Operation Use Cases

Below are a list of common use cases and the configuration required.

Use Cases:
- [Scan a single frequency](#scan-a-single-frequency)
- [Scan across a frequency range](#scan-across-a-frequency-range)
- [Scan across a non-linear range](#scan-across-a-non-linear-range)
- [Deploy signal detector](#deploy-signal-detector)
- [Deploy Birdseye](#deploy-birdseye)
- [Scan a single frequency with model](#scan-a-single-frequency-with-model)
- [Scan across a frequency range with model](#scan-a-single-frequency-with-model)
- [Scan across a non-linear range with model](#scan-across-a-non-linear-range-with-model)
- [Scan across freq with model feeding to Birdseye](#scan-across-freq-with-model-feeding-to-birdseye)

### Scan a single frequency
### Scan across a frequency range
### Scan across a non-linear range
### Deploy signal detector
### Deploy Birdseye
### Scan a single frequency with model
### Scan across a frequency range with model
### Scan across a non-linear range with model
### Scan across freq with model feeding to Birdseye

## Running multiple radios on the same machine

gamutRF supports the use of multiple radios on the same machine, whether for scanning or recording. When using multiple Ettus SDRs, assign a specific radio to a specific container by specifying the ```serial``` UHD driver argument. To list all connected Ettus radios, run ```uhd_find_devices```. Then add the ```--sdrargs``` argument to the specific container. For example, ```--sdrargs num_recv_frames=960,recv_frame_size=16360,type=b200,serial=12345678```.

gamutRF also supports the KrakenSDR (which presents as five librtlsdr radios). You can run a scanner container for each radio, by adding a serial number - for example, ```--sdr=rtlsdr,serial=1000```. Use ```SoapySDRUtil --find``` to check the radios are correctly connected.

## Manually initiating worker actions

The orchestrator has a web interface on port 80. You can use this to command a worker to start an I/Q sample recording or start a RSSI stream.

You can also call the worker API directly (for example, to use a stand-alone worker to make a recording or stream RSSI).

For example, make a recording with a sample rate of 20.48e6, for 2 seconds (409600000 samples worth), at 100MHz:

```
$ wget -nv -O- localhost:8000/v1/record/100000000/409600000/20480000
```

To stream RSSI values instead, call:

```
$ wget -nv -O- localhost:8000/v1/rssi/100000000/409600000/20480000
```

If the sample count parameter is 0, the stream will not end
until a new RPC (whether rssi or record) is received.

## Utility Functions

Below are some utility functions for working with GamutRF and the collected IQ files.

### Running GamutRF pipeline from IQ file

gamutrf-offline can be used to run a previously recorded IQ file through GamutRF to simulate expected behavior of FFT, spectrogram generation, and model performance. To run in offline mode you will need to supply the relevant IQ file as either .sigmf-data file and the relevant .sigmf-meta file or the .zst or raw IQ file with the appropriately formatted naming convention.

```bash
docker run -v $(pwd):/gamutrf/data -ti iqtlabs/gamutrf gamutrf-offline --tune-step-fft=512 --db_clamp_floor=-150 --fft_batch_size=256 --nfft=1024 "data/{source_file}.sigmf-meta"
```

### Working with worker I/Q recordings

Workers make recordings that are compressed with zstandard, and are typically in complex number, int16 format, and include the center frequency and sample rate that the recording was made with. gamutRF tools can generally work with such files directly, but other tools require the recordings to be converted (see below).

### Generating a spectrogram of a recording

gamutRF provides a tool to convert a recording or directory of recordings into a spectrogram. For example, to convert all I/Q recordings in /tmp:

```docker run -ti -v /tmp:/tmp iqtlabs/gamutrf gamutrf-specgram /tmp```

Use the ```--help``` option to change how the spectogram is generated (for example, to change the sample rate).

### Reviewing a recording interactively in gqrx

[gqrx](https://gqrx.dk/) is a multiplatform open source tool that allows some basic SDR operations like visualizing or audio demodulating an I/Q sample recording (see the [github releases page](https://github.com/gqrx-sdr/gqrx/releases), for a MacOS .dmg file). To use gqrx with a gamutRF recording, first translate the recording to gnuradio format (see above). Then open gqrx.

* Select ```Complex Sampled (I/Q) File```
* Set ```Input rate``` to be the same as the gamutRF sample rate (e.g. from the recording file name,
```gamutrf_recording_ettus_directional_gain70_1234_100000000Hz_20971520sps.raw```, set ```Input rate``` to 20971520, and also edit ```rate=``` in ```Device string``` to be 20971520)
* Set ``Bandwidth`` to 0
* Edit ```Device string``` to set the ```file=``` to be the path to the recording.
* Set ```Decimation``` to None.
* Finally select ```OK``` and then ```play``` from the gqrx interface to watch the recording play.

### Demodulating AM/FM audio from a recording

gamutRF provides a tool to demodulate AM/FM audio from a recording as an example use case.

* Use the ```airspyfm``` tool to demodulate audio to a WAV file.

For example, to decode an FM recording which must be at the center frequency of a recording:

```docker run -v /tmp:/tmp -ti iqtlabs/gamutrf-airspyfm -m fm -t filesource -c filename=/tmp/gamutrf_recording_gain70_1234_100000000Hz_2097152sps.raw,raw,format=FLOAT,srate=2097152 -F /tmp/out.wav```

Run:

```docker run -ti iqtlabs/gamutrf-airspyfm -h```

To view other options.

### API access

gamutRF supports two separate APIs - for receiving scanner updates, and making scanner configuration changes.

* scanner update API: described in [zmqreceiver.py](gamutrf/zmqreceiver.py). Receives real time scanner updates and config.
* scanner config API: allows RESTful updates to any CLI argument. Eg, ```wget -O- "http://localhost:9001/reconf?freq_start=1e9&freq_end=2e9"``` causes the scanner to reset and scan 1e9 to 2e9Hz. Any config change causes the scanner's gnuradio flowgraph to restart.
