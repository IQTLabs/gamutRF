#!/bin/bash

set -e

TMPDIR=/tmp
TESTFILE=gamutrf_recording_ettus__gain40_1_10000000Hz_1024000sps.s16
rm -rf "$TMPDIR/input"
mkdir "$TMPDIR/input"
export FULLTMP=$TMPDIR/input/$TESTFILE
python -c "import numpy as np ; x = (np.random.uniform(-16384,16383,(20480000,)) + 1.j * np.random.uniform(-16384,16383,(20480000,))).tobytes() ; np.frombuffer(x, np.float64).astype(np.int16).tofile(\"$FULLTMP\")"

rm -rf "$TMPDIR/ref"
mkdir "$TMPDIR/ref"

run_offline()
{
    dir=$1
    docker run -v "$TMPDIR:/gamutrf/data" -t iqtlabs/gamutrf gamutrf-offline --tune-step-fft=512 --db_clamp_floor=-150 --nfft=1024 --rotate_secs=0 --inference_output_dir=/gamutrf/data/"$dir" /gamutrf/data/input/$TESTFILE
}

run_offline ref

for trial in {1..5} ; do
  rm -rf "$TMPDIR/trial"
  mkdir "$TMPDIR/trial"
  run_offline trial
  for image in $TMPDIR/ref/image*png ; do
    baseimage="$(basename $image)"
    testimage="$TMPDIR/trial/$baseimage"
    diff -b "$image" "$testimage"
  done
done
