FROM ubuntu:20.04
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    apt-get -y --no-install-recommends install \
    cmake \
    g++ \
    git \
    libbladerf-dev \
    libboost-all-dev \
    libfftw3-dev \
    libgmp-dev \
    libgsl-dev \
    liblog4cpp5-dev \
    libpython3-dev \
    libsndfile1-dev \
    libthrift-dev \
    libuhd-dev \
    libunwind-dev \
    libvolk2-dev \
    make \
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
RUN cmake .. && make -j "$(nproc)" && make install && ldconfig -v
WORKDIR /root/gr-habets39
RUN git checkout 9961d0b8644bacbc932e46042a34b4871ba627f8
WORKDIR /root/gr-habets39/build
RUN cmake .. && make install
RUN ln -s /usr/local/lib/python3/dist-packages/* /usr/local/lib/python3.8/dist-packages
RUN ldconfig -v

COPY gamutrf/scan.py /root/scan.py

ENTRYPOINT ["/usr/bin/python3", "/root/scan.py"]
CMD ["--help"]
