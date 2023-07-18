import sys

try:
    from gnuradio import blocks
    from gnuradio import gr
    from gnuradio import network
except ModuleNotFoundError as err:
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme): %s"
        % err
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
        sources, _cmd_port, _workaround_start_hook = get_source(
            args.sdr, samp_rate, args.gain, agc, center_freq
        )

        rssi_blocks = sources + [
            blocks.complex_to_mag_squared(1),
            blocks.moving_average_ff(self.mean_window, 1 / self.mean_window, 2000, 1),
            blocks.nlog10_ff(10, 1, 0),
            blocks.add_const_ff(-34),
            blocks.keep_one_in_n(gr.sizeof_float, int(self.rssi_throttle)),
            network.udp_sink(
                gr.sizeof_float, 1, RSSI_UDP_ADDR, RSSI_UDP_PORT, 0, 32768, False
            ),
        ]

        last_block = rssi_blocks[0]
        for block in rssi_blocks[1:]:
            self.connect((last_block, 0), (block, 0))
            last_block = block
