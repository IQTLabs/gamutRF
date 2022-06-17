import time

from gnuradio import blocks
from gnuradio import gr
from gnuradio import network
from gnuradio import soapy
from gnuradio import uhd

from gamutrf.utils import ETTUS_ANT
from gamutrf.utils import ETTUS_ARGS

FLOAT_SIZE = 4
RSSI_UDP_ADDR = '127.0.0.1'
RSSI_UDP_PORT = 2001
MAX_RSSI = 100


class BirdsEyeRSSI(gr.top_block):

    def __init__(self, args, samp_rate, center_freq, rssi_throttle=10, agc=False):
        gr.top_block.__init__(self, 'BirdsEyeRSSI', catch_exceptions=True)

        self.threshold = args.rssi_threshold
        self.mean_window = args.mean_window
        self.samp_rate = samp_rate
        self.gain = args.gain
        self.center_freq = center_freq
        self.rssi_throttle = rssi_throttle

        dev = f'driver={args.sdr}'
        stream_args = ''
        tune_args = ['']
        settings = ['']

        # TODO: use common code with grscan.py
        if args.sdr == 'ettus':
            self.source_0 = uhd.usrp_source(
                ','.join((ETTUS_ARGS, '')),
                uhd.stream_args(
                    cpu_format='fc32',
                    args='',
                    channels=list(range(0, 1)),
                ),
            )
            self.source_0.set_antenna(ETTUS_ANT, 0)
            self.source_0.set_samp_rate(self.samp_rate)
            self.source_0.set_center_freq(self.center_freq)
            self.source_0.set_gain(self.gain, 0)
            self.source_0.set_rx_agc(agc, 0)
        else:
            self.source_0 = soapy.source(
                dev, 'fc32', 1, '', stream_args, tune_args, settings)
            self.source_0.set_sample_rate(0, self.samp_rate)
            self.source_0.set_bandwidth(0, 0.0)
            self.source_0.set_frequency(0, self.center_freq)
            self.source_0.set_frequency_correction(0, 0)
            self.source_0.set_gain_mode(0, agc)
            self.source_0.set_gain(0, self.gain)

        self.network_udp_sink_0 = network.udp_sink(
            gr.sizeof_float, 1, RSSI_UDP_ADDR, RSSI_UDP_PORT, 0, 32768, False)
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(10, 1, 0)
        self.blocks_moving_average_xx_0 = blocks.moving_average_ff(
            self.mean_window, 1, 2000, 1)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)
        self.blocks_add_const_vxx_0 = blocks.add_const_ff(-60)
        self.keep_one_in_n_0 = blocks.keep_one_in_n(
            gr.sizeof_float, int(self.rssi_throttle))

        self.connect((self.keep_one_in_n_0, 0), (self.network_udp_sink_0, 0))
        self.connect((self.blocks_add_const_vxx_0, 0),
                     (self.keep_one_in_n_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0),
                     (self.blocks_moving_average_xx_0, 0))
        self.connect((self.blocks_moving_average_xx_0, 0),
                     (self.blocks_nlog10_ff_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0),
                     (self.blocks_add_const_vxx_0, 0))
        self.connect((self.source_0, 0),
                     (self.blocks_complex_to_mag_squared_0, 0))
