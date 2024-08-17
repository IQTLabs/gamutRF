# gamutRF

gamutRF is a gnuradio-based SDR-based scanner, I/Q signal collector and identifier (using a image or I/Q based pyTorch model).

While it can run on Pi4 and Pi5 machines (and its components can be distributed over a network), it is more typically deployed on a single x86_64 machine with an nvidia GPU (see [deployment instructions](https://github.com/IQTLabs/gamutrf-deploy)).
 
gamutRF's scanner container connects to a local SDR and sweeps over a configured frequency range or ranges collecting samples. When a configurable number of [valid I/Q samples](https://github.com/IQTLabs/gr-iqtlabs/blob/7990932fa871aa3f84e75771052500551f615638/lib/retune_pre_fft_impl.cc#L276) are received the SDR is retuned to a new interval (see [blocks in gr-iqtlabs](https://github.com/IQTLabs/gr-iqtlabs)). The samples are processed and sent to a waterfall container for display, and optionally to a [Torchserve](https://github.com/IQTLabs/torchserve) instance for identification. Recording, and basic parameters (such as the frequency range to scan) can be controlled from the waterfall container.

## License

Distributed under the [Apache 2.0](./LICENSE). See [LICENSE](./LICENSE) for more information.

## Contact IQTLabs

- Twitter: [@iqtlabs](https://twitter.com/iqtlabs)
- Email: info@iqtlabs.org

See our other projects: [https://github.com/IQTLabs/](https://github.com/IQTLabs/)

## AIR-T support

GamutRF has legacy support for the [Deepwave AIR-T](docs/README-airt.md)

## Development

Development with GamutRF requires familiarity with gnuradio, an SDR, a x86_64 host running Ubuntu 24.04 with Docker installed (and ideally an nvidia GPU, though this is not required).

### Local development

* Install [gnuradio](https://wiki.gnuradio.org/index.php/InstallingGR), 3.10 or later
* Install [gr-iqtlabs](https://github.com/IQTLabs/gr-iqtlabs)
* Make modifications, and install with ```poetry install```
* Run tests with ```pytest```

### Docker development

Follow above local development instructions, and then build containers (tests will be run inside the containers).

* ```docker build -f docker/Dockerfile.base docker -t iqtlabs/gamutrf-base:latest```
* ```docker build -f Dockerfile . -t iqtlabs/gamutrf:latest```
