# Extract and decimate narrowband signal from wideband signal
#
# Example of decoding FM signal centered at 98.1MHz, from 20MHz wideband recording centered at 100MHz recorded at 20Msps.
#
# Convert int16 recording to complex float:
#
# $ samples2raw.py gamutrf_recording1643863029_100000000Hz_20000000sps.s16.gz
#
# Select new center -1.9MHz, from original center and downsample to 1Msps
#
# $ freqxlator.py --samp-rate=20e6 --center=-1.9e6 --dec=20 --infile gamutrf_recording1643863029_100000000Hz_20000000sps.raw --outfile fm.raw
#
# Decode 1Msps recording as FM.
#
# $ airspy-fmradion -t filesource -c srate=1000000,raw,format=FLOAT,filename=fm.raw -W fm.wav
import argparse
import sys

try:
    from gnuradio import blocks  # pytype: disable=import-error
    from gnuradio import eng_notation  # pytype: disable=import-error
    from gnuradio import gr  # pytype: disable=import-error
    from gnuradio.eng_arg import eng_float  # pytype: disable=import-error
    from gnuradio.filter import firdes
    from gnuradio.filter import freq_xlating_fir_filter_ccc
    from pmt import PMT_NIL  # pytype: disable=import-error
except ModuleNotFoundError as err:
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme): %s"
        % err
    )
    sys.exit(1)


class FreqXLator(gr.top_block):
    def __init__(self, samp_rate, center, transitionbw, dec, infile, outfile):
        gr.top_block.__init__(self, "freqxlator", catch_exceptions=True)

        self.samp_rate = samp_rate
        self.center = center
        self.transitionbw = transitionbw
        self.dec = dec
        self.infile = infile
        self.outfile = outfile

        self.freq_xlating_fir_filter_xxx_0 = freq_xlating_fir_filter_ccc(
            self.dec, self._get_taps(), self.center, self.samp_rate
        )
        self.blocks_file_source_0 = blocks.file_source(
            gr.sizeof_gr_complex, self.infile, False, 0, 0
        )  # pylint: disable=no-member
        self.blocks_file_source_0.set_begin_tag(PMT_NIL)
        self.blocks_file_sink_0 = blocks.file_sink(
            gr.sizeof_gr_complex, self.outfile, False
        )  # pylint: disable=no-member
        self.blocks_file_sink_0.set_unbuffered(False)

        self.connect(
            (self.blocks_file_source_0, 0), (self.freq_xlating_fir_filter_xxx_0, 0)
        )
        self.connect(
            (self.freq_xlating_fir_filter_xxx_0, 0), (self.blocks_file_sink_0, 0)
        )

    def _get_taps(self):
        return firdes.complex_band_pass(
            1,
            self.samp_rate,
            -self.samp_rate / (2 * self.dec),
            self.samp_rate / (2 * self.dec),
            self.transitionbw,
        )


def argument_parser():
    parser = argparse.ArgumentParser(
        "Extract and decimate narrowband signal from wideband signal"
    )
    parser.add_argument("infile", type=str, help="Input file (complex I/Q format)")
    parser.add_argument("outfile", type=str, help="Output file (complex I/Q format)")
    parser.add_argument(
        "--samp-rate",
        dest="samp_rate",
        type=eng_float,
        default=eng_notation.num_to_str(float(20e6)),
        help="Set samp_rate [default=%(default)r]",
    )
    parser.add_argument(
        "--center",
        dest="center",
        type=eng_float,
        default=eng_notation.num_to_str(float(-1e6)),
        help="Offset to new center frequency [default=%(default)r]",
    )
    parser.add_argument(
        "--transitionbw",
        dest="transitionbw",
        type=eng_float,
        default=eng_notation.num_to_str(float(10e3)),
        help="Filter transmission bandwidth [default=%(default)r]",
    )
    parser.add_argument(
        "--dec",
        dest="dec",
        type=int,
        default=20,
        help="Decimation [default=%(default)r]",
    )
    return parser


def main():
    parser = argument_parser()
    options = parser.parse_args()
    block = FreqXLator(
        options.samp_rate,
        options.center,
        options.transitionbw,
        options.dec,
        options.infile,
        options.outfile,
    )
    block.start()
    block.wait()
