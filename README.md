# gamutRF

gamutRF is a gnuradio-based SDR-based scanner, I/Q signal collector and identifier (using a image or I/Q based pyTorch model).

While it can run on Pi4 and Pi5 machines (and its components can be distributed over a network), it is more typically deployed on a single x86_64 machine with an nvidia GPU (see [deployment instructions](https://github.com/IQTLabs/gamutrf-deploy)).
 
gamutRF's scanner container connects to a local SDR and sweeps over a configured frequency range or ranges collecting samples. When a configurable number of valid I/Q samples are received the SDR is retuned to a new interval (see [blocks in gr-iqtlabs](https://github.com/IQTLabs/gr-iqtlabs)). The samples are processed and sent to a waterfall container for display, and optionally to a Torchserve instance for identification. Recording, and basic parameters (such as the frequency range to scan) can be controlled from the waterfall container.

## License

Distributed under the [Apache 2.0](./LICENSE). See [LICENSE](./LICENSE) for more information.

## Contact IQTLabs

- Twitter: [@iqtlabs](https://twitter.com/iqtlabs)
- Email: info@iqtlabs.org

See our other projects: [https://github.com/IQTLabs/](https://github.com/IQTLabs/)
