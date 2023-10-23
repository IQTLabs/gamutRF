# *** fork for AIRT support ***

## install prerequisites

Ensure you have the latest docker installed, and also qemu:

```apt-get install -y qemu-user-static```

Build gnuradio with soapy frequency tuning command patches, on x86 host,
if you do not already have the ```anarkiwi-airt-gnuradio``` channel.

```
$ git clone https://github.com/anarkiwi/gnuradio-feedstock -b airt3
$ cd gnuradio-feedstock
$ ./build-locally.py
```

choose option 6 (```linux_aarch64_numpy1.20python3.9.____cpython```).

tar up ```build_artifacts``` and copy into $HOME on AIRT, as ```anarkiwi-airt-gnuradio```.
*NOTE:* ensure you do not have $DISPLAY set during the build to avoid a hang.

Ensure you have airstack 0.5.5 .deb installed (includes tuning speed fixes - neither earlier nor later versions work).

Configure xdma (otherwise tuning will be very slow).

```
$ sudo rmmod xdma
$ sudo modprobe xdma rx_buffer_size_pages=4096
```

Create anarkiwi-airt.yml

```
name: anarkiwi-airt
channels:
  - /home/anarkiwi/anarkiwi-airt-gnuradio
  - file://opt/deepwave/conda-channels/airstack-conda
  - conda-forge
  - nvidia
  - defaults

dependencies:
  - cmake
  - matplotlib
  - numpy
  - opencv
  - pip
  - pybind11
  - python=3.9
  - scipy
  - soapysdr-module-airt=0.5.5
  - gnuradio=3.9.8

  - pip:
    - https://archive.deepwavedigital.com/onnxruntime-gpu/onnxruntime_gpu-1.10.0-cp39-cp39-linux_aarch64.whl
    - https://archive.deepwavedigital.com/pycuda/pycuda-2020.1-cp39-cp39-linux_aarch64.whl
    - https://archive.deepwavedigital.com/tensorrt/tensorrt-8.0.1.6-cp39-none-linux_aarch64.whl
    - https://archive.deepwavedigital.com/pytools/pytools-2022.1.12-py2.py3-none-any.whl
    - --extra-index-url https://pip.cupy.dev/aarch64
    - cupy-cuda102
```

create conda environment

```
$ conda env create -f anarkiwi-airt.yml
```

activate conda

```
$ conda activate anarkiwi-airt
```

ensure libvulkan-dev is not present

```
$ sudo apt-get purge libvulkan-dev
```

install gr-iqtlabs

```
$ git clone https://github.com/google/flatbuffers -b v23.5.26
$ git clone https://github.com/nlohmann/json -b v3.11.2
$ git clone https://github.com/deepsig/libsigmf -b v1.0.2
$ git clone https://github.com/iqtlabs/gr-iqtlabs -b 1.0.45
$ mkdir -p flatbuffers/build && cd flatbuffers/build && cmake -DCMAKE_INSTALL_PREFIX=~/.conda/envs/$CONDA_DEFAULT_ENV .. && make -j $(nproc) && make install && cd ../..
$ mkdir -p json/build && cd json/build && cmake -DCMAKE_INSTALL_PREFIX=~/.conda/envs/$CONDA_DEFAULT_ENV .. && make -j $(nproc) && make install && cd ../..
$ mkdir -p libsigmf/build && cd libsigmf/build && cmake -DUSE_SYSTEM_JSON=ON -DUSE_SYSTEM_FLATBUFFERS=ON -DCMAKE_INSTALL_PREFIX=~/.conda/envs/$CONDA_DEFAULT_ENV -DCMAKE_CXX_FLAGS="-I $HOME/.conda/envs/$CONDA_DEFAULT_ENV/include" .. && make -j $(nproc) && make install && cd ../..
$ mkdir -p gr-iqtlabs/build && cd gr-iqtlabs/build && cmake -DCMAKE_INSTALL_PREFIX=~/.conda/envs/$CONDA_DEFAULT_ENV .. && make -j $(nproc) && make install && cd ../..
```

install gr-wavelearner

```
$ git clone https://github.com/deepwavedigital/gr-wavelearner
$ mkdir -p gr-wavelearner/build && cd gr-wavelearner/build && cmake -DCMAKE_INSTALL_PREFIX=~/.conda/envs/$CONDA_DEFAULT_ENV .. && make -j $(nproc) && make install
```

install gamutrf

```
$ git clone https://github.com/iqtlabs/gamutrf
$ cd gamutrf
$ pip3 install .
```

run gamutrf (may need to change sample rate depending on SDR - e.g. 125e6 or 100e6).

```
$ LD_PRELOAD=$HOME/.conda/envs/$CONDA_DEFAULT_ENV/lib/libgomp.so.1 gamutrf-scan --sdr=SoapyAIRT --freq-start=300e6 --freq-end=6e9 --tune-step-fft 256 --samp-rate=100e6 --nfft 256 --pretune --no-tag-now
```

gamutrf-scan will repeatedly print

```
gr::log :DEBUG: retune_fft0 - rx_freq: {}
gr::log :DEBUG: retune_fft0 - new rx_freq tag: {}, last {}
gr::log :DEBUG: retune_fft0 - retuning to {}
```

while in operation - this is normal, the broken logging is because gamutrf/gr-iqtlabs are written for gnuradio 3.10, not for gnuradio 3.9 style logging.


## inference with torchserve

On a non-AIRT machine that the AIRT can reach over the network, that has an nvidia GPU and docker installed:

# docker nvidia support

See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

# start torchserve

From gamutRF's source directory:

```
$ mkdir /tmp/torchserve
$ cp torchserve/config.properities /tmp/torchserve
$ docker run --gpus all -p 8081:8081 -p 8080:8080 -v /tmp/torchserve:/torchserve -d iqtlabs/gamutrf-cuda-torchserve torchserve --start --model-store /torchserve --ts-config /torchserve/config.properties --ncs --foreground
```

# create and register model

From gamutRF's source directory, and having obtained mini2_snr.pt:

```
$ pip3 install torch-model-archiver
$ torch-model-archiver --force --model-name mini2_snr --version 1.0 --serialized-file /PATH/TO/mini2_snr.pt --handler torchserve/custom_handler.py --export-path /tmp/torchserve
$ curl -X POST "localhost:8081/models?model_name=mini2_snr&url=mini2_snr.mar&initial_workers=4&batch_size=2"
```

Now, when starting the scanner, on the AIRT:

```
$ LD_PRELOAD=$HOME/.conda/envs/$CONDA_DEFAULT_ENV/lib/libgomp.so.1 gamutrf-scan --sdr=SoapyAIRT --freq-start=300e6 --freq-end=6e9 --tune-step-fft 256 --samp-rate=100e6 --nfft 256 --pretune --no-tag-now --inference_model_server TORCHSERVEHOSTNAME:8080 --inference_model_name mini2_snr --inference_output_dir /tmp --inference_min_db -100 --inference_min_confidence 0.5
```

# inference caveats

* torchserve can run in CPU-only mode using the same container if no GPU is available, but inference performance will be poor.
* running torchserve and gamutrf on the same host is not currently recommended (due to CPU resource competition, even when a GPU is available).
