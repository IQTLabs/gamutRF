#!/bin/sh
exec /usr/local/bin/torchserve --start --model-store /model_store --ts-config /torchserve/config.properties --ncs --foreground $*
