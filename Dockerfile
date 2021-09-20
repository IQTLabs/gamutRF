FROM ubuntu:20.04
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:gnuradio/gnuradio-releases && \
    apt-get -y --no-install-recommends install \
    cmake \
    git \
    g++ \
    gnuradio \
    gnuradio-dev \
    make \
    uhd-host && \
    apt -y autoremove --purge && rm -rf /var/cache/* /root/.cache/*

RUN /usr/lib/uhd/utils/uhd_images_downloader.py -t "b2|usb"
WORKDIR /root
RUN git clone https://github.com/ThomasHabets/gr-habets39
WORKDIR /root/gr-habets39
RUN git checkout 9961d0b8644bacbc932e46042a34b4871ba627f8
WORKDIR /root/gr-habets39/build
RUN cmake ..
RUN make && make install
RUN ln -s /usr/local/lib/python3/dist-packages/habets39 /usr/local/lib/python3.8/dist-packages
RUN ldconfig -v

COPY ettus_scan.py /root/ettus_scan.py

ENTRYPOINT ["/root/ettus_scan.py"]
CMD ["--help"]
