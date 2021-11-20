FROM ubuntu:20.04 as builder
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    apt-get -y --no-install-recommends install \
    build-essential \
    cmake \
    g++ \
    gcc \
    git \
    libbladerf-dev \
    libboost-all-dev \
    libev-dev \
    libfftw3-dev \
    libgmp-dev \
    libgsl-dev \
    liblimesuite-dev \
    liblog4cpp5-dev \
    libpython3-dev \
    libncurses5-dev \
    libsndfile1-dev \
    libtecla1 \
    libthrift-dev \
    libuhd-dev \
    libunwind-dev \
    libusb-1.0-0-dev \
    libusb-1.0-0 \
    libvolk2-dev \
    make \
    netcat \
    pkg-config \
    python3-bladerf \
    python3-mako \
    python3-numpy \
    python3-pip \
    python3-pygccxml \
    python3-pytest \
    python3-scipy \
    python3-uhd \
    swig \
    uhd-host \
    wget
RUN /usr/lib/uhd/utils/uhd_images_downloader.py -t "b2|usb"
WORKDIR /root
# https://wiki.gnuradio.org/index.php/GNU_Radio_3.9_OOT_Module_Porting_Guide
RUN wget -q -O- https://github.com/pybind/pybind11/archive/v2.5.0.tar.gz |tar -zxvf -
RUN git clone https://github.com/gnuradio/volk -b v2.5.0
RUN git clone https://github.com/pothosware/SoapySDR -b soapy-sdr-0.8.1
RUN git clone https://github.com/pothosware/SoapyBladeRF -b soapy-bladerf-0.4.1
RUN git clone https://github.com/anarkiwi/gnuradio -b st2
RUN git clone https://github.com/ThomasHabets/gr-habets39
RUN git clone https://github.com/Nuand/bladeRF.git -b 2021.10
RUN git clone https://github.com/anarkiwi/lime-tools -b samples
WORKDIR /root/pybind11-2.5.0/build
RUN cmake .. && make && make install
WORKDIR /root/volk/build
RUN git submodule update --init
RUN cmake .. && make -j "$(nproc)" && make install
WORKDIR /root/SoapySDR/build
RUN cmake .. && make -j "$(nproc)" && make install
WORKDIR /root/SoapyBladeRF/build
RUN cmake .. && make -j "$(nproc)" && make install
WORKDIR /root/gnuradio/build
RUN cmake -DENABLE_POSTINSTALL=OFF -DENABLE_GR_FEC=OFF -DENABLE_GR_CTRLPORT=OFF -DENABLE_GR_DIGITAL=OFF -DENABLE_GR_AUDIO=OFF -DENABLE_GR_CHANNELS=OFF -DENABLE_GR_TRELLIS=OFF -DENABLE_GR_UTILS=OFF -DENABLE_GR_BLOCKTOOL=OFF -DENABLE_GR_VOCODER=OFF -DENABLE_GR_WAVELET=OFF -DENABLE_GR_NETWORK=OFF .. && make -j "$(nproc)" && make install && ldconfig -v
WORKDIR /root/gr-habets39
RUN git checkout 9961d0b8644bacbc932e46042a34b4871ba627f8
WORKDIR /root/gr-habets39/build
RUN cmake .. && make install
WORKDIR /root/bladeRF/host/build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DINSTALL_UDEV_RULES=ON -DENABLE_BACKEND_LIBUSB=TRUE .. && make -j "$(nproc)" && make install && ldconfig -v
WORKDIR /root/lime-tools/build
RUN cmake .. && make install
RUN ln -s /usr/local/lib/python3/dist-packages/* /usr/local/lib/python3.8/dist-packages
RUN ldconfig -v

#FROM ubuntu:20.04
LABEL maintainer="Charlie Lewis <clewis@iqt.org>"
#ENV DEBIAN_FRONTEND noninteractive
ENV UHD_IMAGES_DIR /usr/share/uhd/images
#COPY --from=builder /usr/local /usr/local
#COPY --from=builder /usr/lib/*-linux-gnu /usr/lib/
#RUN apt-get update && apt-get install --no-install-recommends -yq \
#    python3-pip

COPY gamutrf/scan.py /root/scan.py

ENTRYPOINT ["/usr/bin/python3", "/root/scan.py"]
CMD ["--help"]
