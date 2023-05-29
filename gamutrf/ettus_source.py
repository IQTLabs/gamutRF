import time
from gamutrf.utils import ETTUS_ANT
from gamutrf.utils import ETTUS_ARGS


def get_ettus_source(sdrargs, samp_rate, center_freq, agc, gain, uhd):
    if not sdrargs:
        sdrargs = ETTUS_ARGS
    source = uhd.usrp_source(
        ",".join((sdrargs, "")),
        uhd.stream_args(
            cpu_format="fc32",
            args="",
            channels=list(range(0, 1)),
        ),
    )
    source.set_antenna(ETTUS_ANT, 0)
    source.set_samp_rate(samp_rate)
    if center_freq is not None:
        source.set_center_freq(center_freq, 0)
    source.set_gain(gain, 0)
    source.set_rx_agc(agc, 0)

    now = time.time()
    now_sec = int(now)
    now_frac = now - now_sec
    uhd_now = uhd.time_spec(now_sec, now_frac)
    source.set_time_now(uhd_now, uhd.ALL_MBOARDS)
    time_diff = source.get_time_now().get_real_secs() - now
    if time_diff > 5:
        raise ValueError("could not set Ettus SDR time successfully")
    return [source]
