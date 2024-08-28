import logging
import numpy as np
import sys

from scipy import signal

try:
    from gnuradio import gr  # pytype: disable=import-error

    # from cupyx.scipy import signal as cupy_signal

except (ModuleNotFoundError, ImportError) as err:  # pragma: no cover
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme): %s"
        % err
    )
    sys.exit(1)


class dc_spike_detrend(gr.sync_block):
    """
    docstring for block dc_spike_detrend
    """

    def __init__(self, length=1024):
        gr.sync_block.__init__(
            self, name="dc_spike_detrend", in_sig=[np.complex64], out_sig=[np.complex64]
        )
        self.length = length

    def work(self, input_items, output_items):
        signal_input = input_items[0]

        # output_items[0][:] = signal_input - cupy.mean(signal_input)
        output_items[0][:] = signal.detrend(
            signal_input, type="linear", bp=np.arange(0, len(signal_input), self.length)
        )
        # output_items[0][:] = cupy_signal.detrend(signal_input, type="linear", bp=np.arange(0, len(signal_input), self.length)).get()

        return len(signal_input)


class dc_spike_remove(gr.sync_block):
    """
    docstring for block dc_spike_remove
    """

    def __init__(self, ratio=0.995):
        gr.sync_block.__init__(
            self, name="dc_spike_remove", in_sig=[np.complex64], out_sig=[np.complex64]
        )
        self.ratio = ratio
        self.d_avg_real = 0
        self.d_avg_img = 0

    def work(self, input_items, output_items):
        signal_input = input_items[0]
        output = output_items[0]

        for i in range(len(signal_input)):
            self.d_avg_real = (
                self.ratio * (signal_input[i].real - self.d_avg_real) + self.d_avg_real
            )
            self.d_avg_img = (
                self.ratio * (signal_input[i].imag - self.d_avg_img) + self.d_avg_img
            )

            output[i] = np.complex64(
                complex(
                    real=signal_input[i].real - self.d_avg_real,
                    imag=signal_input[i].imag - self.d_avg_img,
                )
            )

        return len(signal_input)
