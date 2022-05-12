#!/usr/bin/env python3

import argparse
import json
import socket
import struct
import time
from gnuradio import analog, blocks, gr, network, soapy
import paho.mqtt.client as mqtt

FLOAT_SIZE = 4
RSSI_UDP_ADDR = '127.0.0.1'
RSSI_UDP_PORT = 2001
MIN_RSSI = -100
MAX_RSSI = 100


class birds_eye_rssi(gr.top_block):

    def __init__(self, args):
        gr.top_block.__init__(self, 'birds_eye_rssi', catch_exceptions=True)

        self.threshold = args.rssi_threshold
        self.samp_rate = args.samp_rate
        self.gain = args.gain
        self.center_freq = args.center_freq

        # TODO: support osmocom if needed
        dev = f'driver={args.sdr}'
        stream_args = ''
        tune_args = ['']
        settings = ['']
        self.source_0 = soapy.source(dev, 'fc32', 1, '', stream_args, tune_args, settings)
        self.source_0.set_sample_rate(0, self.samp_rate)
        self.source_0.set_bandwidth(0, 0.0)
        self.source_0.set_frequency(0, self.center_freq)
        self.source_0.set_frequency_correction(0, 0)
        self.source_0.set_gain(0, min(max(self.gain, -1.0), 60.0))

        self.network_udp_sink_0 = network.udp_sink(gr.sizeof_float, 1, RSSI_UDP_ADDR, RSSI_UDP_PORT, 0, 1472, False)
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(1, 1, 0)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_ff(10)
        self.blocks_moving_average_xx_0 = blocks.moving_average_ff(256, 1, 4000, 1)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)
        self.blocks_add_const_vxx_0 = blocks.add_const_ff(-60)
        self.analog_pwr_squelch_xx_0 = analog.pwr_squelch_cc(self.threshold, 5e-4, 1000, True)

        self.connect((self.analog_pwr_squelch_xx_0, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self.blocks_add_const_vxx_0, 0), (self.network_udp_sink_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_moving_average_xx_0, 0))
        self.connect((self.blocks_moving_average_xx_0, 0), (self.blocks_nlog10_ff_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.blocks_add_const_vxx_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.source_0, 0), (self.analog_pwr_squelch_xx_0, 0))


def report_rssi(args, reported_rssi, reported_time):
    print(f'reporting RSSI {reported_rssi}')
    try:
        client = mqtt.Client()
        client.connect(args.mqqt_server)
        client.loop_start()
        report = {
            'time': reported_time,
            'rssi': reported_rssi,
        }
        client.publish('gamutrf/rssi', json.dumps(report))
        client.loop_stop()
    except (ConnectionRefusedError, mqtt.WebsocketConnectionError, ValueError) as err:
        print(f'failed to report RSSI to MQQT: {err}')


def process_rssi(args, sock):
    last_rssi_time = 0
    while True:
        rssi_raw, _ = sock.recvfrom(FLOAT_SIZE)
        rssi = struct.unpack('f', rssi_raw)[0]
        if rssi < MIN_RSSI or rssi > MAX_RSSI:
            continue
        now = time.time()
        now_diff = now - last_rssi_time
        if now_diff < args.rssi_interval:
            continue
        report_rssi(args, rssi, now)
        last_rssi_time = now


def serve_rssi(args):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((RSSI_UDP_ADDR, RSSI_UDP_PORT))
        process_rssi(args, sock)


def main():
    parser = argparse.ArgumentParser(description='report RSSI for a specified frequency')
    parser.add_argument('--sdr', default='bladerf', type=str,
                        help='SOAPy SDR driver to use')
    parser.add_argument('--samp_rate', default=int(20500000), type=int,
                        help='sample rate in samples per second')
    parser.add_argument('--gain', default=int(45), type=int,
                        help='gain in dB')
    parser.add_argument('--center_freq', default=int(5745000000), type=int,
                        help='center frequency in Hz')
    parser.add_argument('--mqqt_server', default='127.0.0.01', type=str,
                         help='MQQT server to report RSSI')
    parser.add_argument('--rssi_interval', default=1.0, type=float,
                         help='rate limit in seconds for RSSI updates')
    parser.add_argument('--rssi_threshold', default=-45, type=float,
                         help='RSSI reporting threshold')
    args = parser.parse_args()

    rssi_server = birds_eye_rssi(args)
    rssi_server.start()
    serve_rssi(args)


if __name__ == '__main__':
    main()
