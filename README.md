# gamutRF

An SDR orchestrated scanner and collector.

gamutRF is a system enabling a compact network of one or more modest machines (such as Pi4s), each with their own USB SDR (such as an Ettus 
B200mini or a BladeRF XA9), to operate collectively as a configurable wideband scanner and I/Q sample recorder. 

A gamutRF "orchestrator" machine can scan 0.1GHz to 6GHz in 30 seconds to identify signals, and then command potentially many gamutRF "workers" to make I/Q sample recordings for later analysis. 

gamutRF provides tools to work with I/Q sample recordings, and to also record GPS location/compass metadata for the system itself. gamutRF typically runs on networks of Raspberry Pi4s, but can also run on x86 machines, and is based on gnuradio.

See also [instructions on how to build a gamutRF system](BUILD.md).

## Operating gamutRF

See the [build doc](BUILD.md)

### Manually initiating worker actions

The orchestrator has a web interface on port 9000. You can use this to command a worker to start an I/Q sample recording or start a RSSI stream.

## Working with worker I/Q recordings

Workers make recordings that are compressed with zstandard, and are typically in complex number, int16 format, and include the center frequency and sample rate that the recording was made with. gamutRF tools can generally work with such files directly, but other tools require the recordings to be converted (see below). 

### Generating a spectrogram of a recording

gamutRF provides a tool to convert a recording or directory of recordings into a spectrogram. For example, to convert all I/Q recordings in /tmp:

```docker run -ti iqtlabs/gamutrf-specgram -v /tmp:/tmp /tmp```

Use the ```--help``` option to change how the spectogram is generated (for example, to change the sample rate).

### Translating recordings to "gnuradio" format

Most SDR tools by convention take an uncompressed raw binary file as input, of type complex, float32. The user must separately specify to the each SDR tool what sample rate the file was made at to correctly process it. gamutRF provides a tool that converts a compressed recording to an uncompressed binary file. For example:

```
doker run -v /tmp:/tmp -ti iqtlabs/gamutrf-samples2raw /tmp/gamutrf_recording_ettus_directional_gain70_1234_100000000Hz_20971520sps.s16.zst
```

### Reviewing a recording interactively in gqrx

[gqrx](https://gqrx.dk/) is a multiplatform open source tool that allows some basic SDR operations like visualizing or audio demodulating an I/Q sample recording. To use gqrx with a gamutRF recording, first translate the recording to gnuradio format (see above). Then open gqrx.

* Select ```Complex Sampled (I/Q) File``` 
* Set input rate and bandwidth to be the same as the gamutRF sample rate (e.g. from the recording file name, 
```gamutrf_recording_ettus_directional_gain70_1234_100000000Hz_20971520sps.raw```, set input rate and bandwidth to 20971520)
* Edit ```Device string``` to set the ```path=``` to be the path to the recording. 
* Set ```Decimation``` to None. 
* Finally select ```OK``` and then ```play``` from the gqrx interface to watch the recording play.

### Reducing recording sample rate

You may want to reduce the sample rate of a recording (e.g. to use another demodulator tool that doesn't support a high sample rate). gamutRF provides the ```freqxlator``` tool which does this. 

* Translate your gamutRF recording to gnuradio format (see above). 
* Use ```freqxlator``` to create a new recording at a lower sample rate, potentially with a different center frequency. 

For example, to reduce a recording make with gamutRF's default sample rate to 1/10th the rate while leaving the center frequency unchanged, use:

```docker run -ti iqtlabs/gamutrf-freqxlator --samp-rate 20971520 --center 0 --dec 10 gamutrf_recording_gain70_1234_100000000Hz_20971520sps.raw gamutrf_recording_gain70_1234_100000000Hz_2097152sps.raw```

### Demodulating AM/FM audio from a recording

gamutRF provides a tool to demodulate AM/FM audio from a recording as an example use case.

* Use the ```freqxlator``` tool to make new recording at no more than 1Msps. 
* Use the ```airspyfm``` tool to demodulate audio to a WAV file.

For example, to decode an FM recording that is at the center frequency of a recording:

```docker run -v /tmp:/tmp -ti iqtlabs/gamutrf-airspyfm -m fm -t filesource -c filename=/tmp/gamutrf_recording_gain70_1234_100000000Hz_2097152sps.raw,raw,format=FLOAT,srate=2097152 -F /tmp/out.wav```

Run:

```docker run -ti iqtlabs/gamutrf-airspyfm -h```

To view other options.

## Troubleshooting

#### Containers won't start using Ettus SDRs

You may see ```[ERROR] [USB] USB open failed: insufficient permissions``` on initial startup with Ettus SDRs. These devices download firmware and switch USB identities when first powered up. Restart the affected container to work around this.

#### "O"s or warnings about overflows in SDR containers

* Ensure your hardware can support the I/Q sample rate you have configured (gamutRF has been tested on Pi4 at 20Msps, which is the default recording rate). Also ensure your recording medium (e.g. flash drive, USB hard disk) is not timing out or blocking.
* If using a Pi4, make sure you are using active cooling and an adequate power supply (no CPU throttling), and you are using a "blue" USB3 port.
