import logging
import os
import sys
import time
from urllib.parse import urlparse

try:
    from gnuradio import blocks
    from gnuradio import soapy
    from gnuradio import uhd
    from gnuradio.gr import sizeof_gr_complex
except ModuleNotFoundError:  # pragma: no cover
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme)"
    )
    sys.exit(1)

from gamutrf.utils import ETTUS_ANT
from gamutrf.utils import ETTUS_ARGS


def get_source(
    grblock, sdr, samp_rate, gain, agc=False, center_freq=None, sdrargs=None
):
    logging.info(
        f"initializing SDR {sdr} with sample rate {samp_rate}, gain {gain}, agc {agc}"
    )

    url = urlparse(sdr)
    if url.scheme:
        if url.scheme == "file" and os.path.exists(url.path):
            grblock.recording_source_0 = blocks.file_source(
                sizeof_gr_complex, url.path, True, 0, 0
            )
            grblock.source_0 = blocks.throttle(sizeof_gr_complex, samp_rate, True)
            grblock.connect((grblock.recording_source_0, 0), (grblock.source_0, 0))
            # TODO: enable setting frequency change tags on the stream, so can test scanner.
            grblock.freq_setter = lambda _x, _y: None
        else:
            raise ValueError("unsupported/missing file location")
        return

    if sdr == "ettus":
        if not sdrargs:
            sdrargs = ETTUS_ARGS
        grblock.source_0 = uhd.usrp_source(
            ",".join((sdrargs, "")),
            uhd.stream_args(
                cpu_format="fc32",
                args="",
                channels=list(range(0, 1)),
            ),
        )
        grblock.source_0.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)
        grblock.source_0.set_antenna(ETTUS_ANT, 0)
        grblock.source_0.set_samp_rate(samp_rate)
        if center_freq is not None:
            grblock.source_0.set_center_freq(center_freq, 0)
        grblock.source_0.set_gain(gain, 0)
        grblock.source_0.set_rx_agc(agc, 0)
        grblock.freq_setter = lambda x, y: x.set_center_freq(y, 0)
        return

    dev = f"driver={sdr}"
    stream_args = ""
    tune_args = [""]
    settings = [""]
    if sdrargs:
        settings = sdrargs
    grblock.source_0 = soapy.source(
        dev, "fc32", 1, "", stream_args, tune_args, settings
    )
    grblock.source_0.set_sample_rate(0, samp_rate)
    grblock.source_0.set_bandwidth(0, samp_rate)
    if center_freq is not None:
        grblock.source_0.set_frequency(0, center_freq)
    grblock.source_0.set_frequency_correction(0, 0)
    grblock.source_0.set_gain_mode(0, agc)
    grblock.source_0.set_gain(0, gain)
    grblock.freq_setter = lambda x, y: x.set_frequency(0, y)
    return
