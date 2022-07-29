import sys

try:
    from gnuradio import blocks
    from gnuradio import gr
    from gnuradio import network
except ModuleNotFoundError:
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme)"
    )
    sys.exit(1)

from gamutrf.grsource import get_source

FLOAT_SIZE = 4
RSSI_UDP_ADDR = "127.0.0.1"
RSSI_UDP_PORT = 2001
MAX_RSSI = 100


class BirdsEyeRSSI(gr.top_block):
    def __init__(self, args, samp_rate, center_freq, rssi_throttle=10, agc=False):
        gr.top_block.__init__(self, "BirdsEyeRSSI", catch_exceptions=True)

        self.threshold = args.rssi_threshold
        self.mean_window = args.mean_window
        self.rssi_throttle = rssi_throttle
        get_source(self, args.sdr, samp_rate, args.gain, agc, center_freq)

        self.network_udp_sink_0 = network.udp_sink(
            gr.sizeof_float, 1, RSSI_UDP_ADDR, RSSI_UDP_PORT, 0, 32768, False
        )
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(10, 1, 0)
        self.blocks_moving_average_xx_0 = blocks.moving_average_ff(
            self.mean_window, 1 / self.mean_window, 2000, 1
        )
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)
        self.blocks_add_const_vxx_0 = blocks.add_const_ff(-34)
        self.keep_one_in_n_0 = blocks.keep_one_in_n(
            gr.sizeof_float, int(self.rssi_throttle)
        )

        self.connect((self.keep_one_in_n_0, 0), (self.network_udp_sink_0, 0))
        self.connect((self.blocks_add_const_vxx_0, 0), (self.keep_one_in_n_0, 0))
        self.connect(
            (self.blocks_complex_to_mag_squared_0, 0),
            (self.blocks_moving_average_xx_0, 0),
        )
        self.connect((self.blocks_moving_average_xx_0, 0), (self.blocks_nlog10_ff_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.blocks_add_const_vxx_0, 0))
        self.connect((self.source_0, 0), (self.blocks_complex_to_mag_squared_0, 0))
