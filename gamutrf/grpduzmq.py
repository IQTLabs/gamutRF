#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import sys
import pmt
import zmq

try:
    from gnuradio import gr  # pytype: disable=import-error
except ModuleNotFoundError as err:  # pragma: no cover
    print(
        "Run from outside a supported environment, please run via Docker (https://github.com/IQTLabs/gamutRF#readme): %s"
        % err
    )
    sys.exit(1)

DELIM = "\n"


# It would be ideal to just use gnuradio's https://github.com/gnuradio/gnuradio/blob/main/gr-zeromq/lib/pub_msg_sink_impl.cc
# block. Unfortunately, this block calls pmt::serialize(), but we just want to send simple json strings. That means
# a receiver client would need to run pmt::deserialize() which requires an installation of gnuradio.
class pduzmq(gr.basic_block):
    def __init__(
        self,
        zmq_addr,
    ):
        gr.basic_block.__init__(
            self,
            name="pduzmq",
            in_sig=None,
            out_sig=None,
        )
        self.zmq_context = zmq.Context()
        self.zmq_pub = self.zmq_context.socket(zmq.PUB)
        self.zmq_pub.setsockopt(zmq.SNDHWM, 100)
        self.zmq_pub.setsockopt(zmq.SNDBUF, 65536)
        self.zmq_pub.bind(zmq_addr)
        self.message_port_register_in(pmt.intern("json"))
        self.set_msg_handler(pmt.intern("json"), self.receive_pdu)

    def stop(self):
        self.zmq_pub.close()

    def receive_pdu(self, pdu):
        item = pmt.to_python(pmt.cdr(pdu)).tobytes().decode("utf8").strip()
        try:
            self.zmq_pub.send_string(item + DELIM, flags=zmq.NOBLOCK)
        except zmq.ZMQError as e:
            logging.error(str(e))
