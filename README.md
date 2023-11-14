# gamutRF

An SDR orchestrated scanner and collector.

gamutRF is a system enabling a compact network of one or more modest machines (such as Pi4s), each with their own USB SDR (such as an Ettus
B200mini or a BladeRF XA9), to operate collectively as a configurable wideband scanner and I/Q sample recorder.

A gamutRF system comprises an "orchestrator" machine which typically runs at least one scanner service (typically scanning 0.1GHz to 6GHz in 30s) and the sigfinder service, which then
can command potentially many gamutRF "workers" to make I/Q sample recordings for later analysis. sigfinder can command multiple scanners which can be on the same machine or be connected
over a network.

gamutRF provides tools to work with I/Q sample recordings, and to also record GPS location/compass metadata for the system itself. gamutRF typically runs on networks of Raspberry Pi4s, but can also run on x86 machines, and is based on gnuradio.

See also [instructions on how to build a gamutRF system](docs/BUILD.md).

## Scanner theory of operation

gamutRF's scanner container connects to a local SDR and sweeps over a configured frequency range or ranges collecting samples. The samples are sent an FFT block, and then to a [streaming retuning gnuradio block](https://github.com/iqtlabs/gr-iqtlabs), which aggregates and then serves the frequency-annotated FFT points as JSON objects over ZMQ to the "sigfinder" container (see below) and also commands the source SDR block to retune to a new frequency in the range when enough FFT points have been accumulated.

The sigfinder container consumes these FFT points over ZMQ (potentially from many scanners), does some noise processing (correcting FFT points to be in frequency order, computing mean power over 10kHz, and then a rolling mean over 1MHz) and then submits them to [scipy.signals.find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html).

If workers have been provisioned, the sigfinder will then command the workers to make an approximately 10 second I/Q recording at approximately 20Msps of each signal. Each signal peak is assigned a 20MHz bin, which means that if a signal is repeatedly detected with some frequency variation, the assigned recording bin will be constant, and if multiple signals are detected within 20MHz they can be collected simultaneously. A worker by default records at a higher sample rate than the bin size, so that 20MHz signal margins can be recorded.

As there will almost certainly be more signals than workers available, sigfinder will prioritize signals that it least often observed over a configurable number of scanner cycles. It is possible to configure this to `off` so that the recording choice will be random. It is also possible to configure the workers to tell the sigfinder to exclude that worker from certain frequency ranges (if for example the worker SDR cannot handle some part of the frequency spectrum scanned).

## Build and Operating gamutRF

See the [build doc](docs/2-BUILD.md) for instruction on setting up a GamutRF system. Next see the [operating instructions](docs/4-OPERATION.md) for instructions on operating GamutRF.

## Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b dev`)
3. Commit your Changes (`git commit -m 'adding some feature'`)
4. Run (and make sure they pass):

```
black --diff --check gamutrf
```

5. Push to the Branch (`git push origin dev`)
6. Open a Pull Request

See `CONTRIBUTING.md` for more information.

## License

Distributed under the [Apache 2.0](https://github.com/IQTLabs/aisonobuoy-data-science/blob/main/LICENSE.txt). See [LICENSE](./LICENSE) for more information.

## Contact IQTLabs

- Twitter: [@iqtlabs](https://twitter.com/iqtlabs)
- Email: info@iqtlabs.org

See our other projects: [https://github.com/IQTLabs/](https://github.com/IQTLabs/)
