def get_soapy_source(sdr, sdrargs, samp_rate, center_freq, agc, gain, soapy):
    dev = f"driver={sdr}"
    stream_args = ""
    tune_args = [""]
    settings = [""]
    if sdrargs:
        settings = sdrargs.split(",")
    source = soapy.source(dev, "fc32", 1, "", stream_args, tune_args, settings)
    source.set_sample_rate(0, samp_rate)
    source.set_bandwidth(0, samp_rate)
    if center_freq is not None:
        source.set_frequency(0, center_freq)
        source.set_frequency_correction(0, 0)
        source.set_gain_mode(0, agc)
        source.set_gain(0, gain)
    cmd_port = "cmd"
    return [source], cmd_port
