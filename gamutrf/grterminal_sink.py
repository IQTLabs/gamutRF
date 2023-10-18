#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import numpy as np

try:
    from gnuradio import gr  # pytype: disable=import-error
except ModuleNotFoundError as err:  # pragma: no cover
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme): %s"
        % err
    )
    sys.exit(1)


class terminal_sink(gr.sync_block):
    # Prints output layer of classifier to terminal for troubleshooting
    def __init__(self, input_vlen, batch_size):
        self.input_vlen = input_vlen
        self.batch_size = batch_size
        gr.sync_block.__init__(
            self,
            name="terminal_sink",
            in_sig=[(np.float32, self.input_vlen)],
            out_sig=None,
        )
        self.batch_ctr = 0

    def work(self, input_items, output_items):
        in0 = input_items[0]
        _batch = in0.reshape(self.batch_size, -1)
        self.batch_ctr += 1
        return len(input_items[0])
