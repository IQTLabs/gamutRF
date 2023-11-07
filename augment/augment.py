#!/usr/bin/python3

from argparse import ArgumentParser
import os
import sigmf
from sigmf import SigMFFile
from sigmf.utils import get_data_type_str
import numpy as np
import torchsig.transforms.transforms as ST
from torchsig.transforms.functional import (
    to_distribution,
    uniform_continuous_distribution,
    uniform_discrete_distribution,
)
from torchsig.utils.types import SignalData, SignalDescription
from gamutrf.sample_reader import read_recording
from gamutrf.waterfall_samples import parse_filename


def make_signal(samples, sample_rate, center_frequency):
    num_iq_samples = samples.shape[0]
    desc = SignalDescription(
        sample_rate=sample_rate,
        num_iq_samples=num_iq_samples,
        center_frequency=center_frequency,
    )
    # TODO: subclass SignalData with alternate constructor that can take just numpy array
    signal = SignalData(samples.tobytes(), np.float32, np.complex128, desc)
    return signal


def get_nosigmf_file(filename):
    meta = parse_filename(filename)
    sample_rate = meta["sample_rate"]
    sample_dtype = meta["sample_dtype"]
    sample_len = meta["sample_len"]
    center_frequency = meta["freq_center"]
    samples = None
    for samples_buffer in read_recording(
        filename, sample_rate, sample_dtype, sample_len, max_sample_secs=None
    ):
        if samples is None:
            samples = samples_buffer
        else:
            samples = np.concatenate([samples, samples_buffer])
    signal = make_signal(samples, sample_rate, center_frequency)
    return filename, signal


def get_signal(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    meta_ext = filename.find(".sigmf-meta")
    if meta_ext == -1:
        return get_nosigmf_file(filename)

    meta = sigmf.sigmffile.fromfile(filename)
    data_filename = filename[:meta_ext]
    meta.set_data_file(data_filename)
    # read_samples() always converts to host cf32.
    samples = meta.read_samples()
    global_meta = meta.get_global_info()
    sample_rate = global_meta["core:sample_rate"]
    sample_type = global_meta["core:datatype"]
    captures_meta = meta.get_captures()
    center_frequency = None
    if captures_meta:
        center_frequency = captures_meta[0].get("core:frequency", None)
    signal = make_signal(samples, sample_rate, center_frequency)
    return data_filename, signal


def write_signal(filename, signal, transforms_text):
    first_desc = signal.signal_description[0]
    signal.iq_data = signal.iq_data.astype(np.complex64)
    signal.iq_data.tofile(filename)

    new_meta = SigMFFile(
        data_file=filename,
        global_info={
            SigMFFile.DATATYPE_KEY: get_data_type_str(signal.iq_data),
            SigMFFile.SAMPLE_RATE_KEY: first_desc.sample_rate,
            SigMFFile.VERSION_KEY: sigmf.__version__,
            SigMFFile.DESCRIPTION_KEY: transforms_text,
        },
    )
    new_meta.add_capture(
        0,
        metadata={
            SigMFFile.FREQUENCY_KEY: first_desc.center_frequency,
        },
    )
    new_meta.tofile(".".join([filename, "sigmf-meta"]))


def augment(signal, filename, output_dir, n, transforms_text):
    # TODO: sadly, due to Torchsig complexity, literal_eval can't be used.
    transforms = eval(transforms_text) # nosec
    i = 0
    base_augment_name = os.path.basename(filename)
    dot = base_augment_name.find(".")
    if dot != -1:
        base_augment_name = base_augment_name[:dot]
    for _ in range(n):
        while True:
            augment_name = os.path.join(
                output_dir, f"augmented-{i}-{base_augment_name}"
            )
            if not os.path.exists(augment_name):
                break
            i += 1
        new_signal = transforms(signal)
        write_signal(augment_name, new_signal, transforms_text)


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "filename",
        type=str,
        help="sigMF file or gamutRF zst recording",
    )
    parser.add_argument("outdir", type=str, help="output directory")
    parser.add_argument("n", type=int, help="number of augmentation passes")
    parser.add_argument(
        "transforms",
        type=str,
        help="transforms to eval, e.g. ST.Compose([ST.AddNoise((-40, -20)),ST.RandomPhaseShift(uniform_continuous_distribution(-1, 1))]) (use quotes)",
    )
    return parser


def main():
    options = argument_parser().parse_args()
    data_filename, signal = get_signal(options.filename)
    augment(signal, data_filename, options.outdir, options.n, options.transforms)


if __name__ == "__main__":
    main()
