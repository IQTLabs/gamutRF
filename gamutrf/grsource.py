import logging
import time

from gnuradio import soapy
from gnuradio import uhd

from gamutrf.utils import ETTUS_ANT
from gamutrf.utils import ETTUS_ARGS


def get_source(sdr, samp_rate, gain, agc=False, center_freq=None):
    stream_args = ''
    tune_args = ['']
    settings = ['']
    logging.info(
        f'initializing SDR {sdr} with sample rate {samp_rate}, gain {gain}, agc {agc}')

    if sdr == 'ettus':
        source_0 = uhd.usrp_source(
            ','.join((ETTUS_ARGS, '')),
            uhd.stream_args(
                cpu_format='fc32',
                args='',
                channels=list(range(0, 1)),
            ),
        )
        source_0.set_time_now(
            uhd.time_spec(time.time()), uhd.ALL_MBOARDS)
        source_0.set_antenna(ETTUS_ANT, 0)
        source_0.set_samp_rate(samp_rate)
        if center_freq is not None:
            source_0.set_center_freq(center_freq, 0)
        source_0.set_gain(gain, 0)
        source_0.set_rx_agc(agc, 0)
        def freq_setter(x, y): return x.set_center_freq(y, 0)
        return (source_0, freq_setter)

    dev = f'driver={sdr}'
    try:
        source_0 = soapy.source(
            dev, 'fc32', 1, '', stream_args, tune_args, settings)
    except RuntimeError:
        return (None, None)
    source_0.set_sample_rate(0, samp_rate)
    source_0.set_bandwidth(0, samp_rate)
    if center_freq is not None:
        source_0.set_frequency(0, center_freq)
    source_0.set_frequency_correction(0, 0)
    source_0.set_gain_mode(0, agc)
    source_0.set_gain(0, gain)
    def freq_setter(x, y): return x.set_frequency(0, y)
    return (source_0, freq_setter)
