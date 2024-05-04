#!/bin/bash

set -e

TMPDIR=/tmp
TESTFILE=gamutrf_recording_ettus__gain40_1_10000000Hz_1024000sps.raw
rm -rf "$TMPDIR/input"
mkdir "$TMPDIR/input"
export FULLTMP=$TMPDIR/input/$TESTFILE
python -c "import numpy as np ; (np.random.uniform(0, 1, 10240000) + 1.j * np.random.uniform(0, 1, 10240000)).astype(np.complex64).tofile(\"$FULLTMP\")"

rm -rf "$TMPDIR/ref"
mkdir "$TMPDIR/ref"

run_offline()
{
    dir=$1
    docker run -v "$TMPDIR:/gamutrf/data" -e LP_NATIVE_VECTOR_WIDTH=128 -t iqtlabs/gamutrf gamutrf-offline --tune-step-fft=512 --db_clamp_floor=-150 --nfft=1024 --rotate_secs=0 --inference_output_dir=/gamutrf/data/"$dir" --sample_dir="/gamutrf/data/samples" --write_samples=1000000000 /gamutrf/data/input/$TESTFILE
}

run_offline ref

sudo zstd -d $TMPDIR/samples/samples*zst
OUTSIZE=$(stat -c%s $TMPDIR/samples/samples*raw)
echo truncating to $OUTSIZE
sudo truncate --size=$OUTSIZE $FULLTMP
diff -b $TMPDIR/samples/samples*.raw $FULLTMP
ech OK
