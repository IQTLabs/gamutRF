import argparse
import datetime
import os
import re
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from gamutrf.utils import is_fft
from gamutrf.sample_reader import get_reader
from gamutrf.utils import SAMPLE_DTYPES, SAMPLE_FILENAME_RE

FFT_FILENAME_RE = re.compile(
    r"^.+_([0-9]+)_([0-9]+)points_([0-9]+)Hz_([0-9]+)sps\.(s\d+|raw).*$"
)


def parse_filename(filename):
    # FFT is always float not matter the original sample type.
    if is_fft(filename):
        sample_type = "raw"
        match = FFT_FILENAME_RE.match(filename)
        try:
            timestamp = int(match.group(1))
            nfft = int(match.group(2))
            freq_center = int(match.group(3))
            sample_rate = int(match.group(4))
            # sample_type = match.group(3)
        except AttributeError:
            timestamp = None
            nfft = None
            freq_center = None
            sample_rate = None
            sample_type = None
    else:
        match = SAMPLE_FILENAME_RE.match(filename)
        nfft = None
        try:
            timestamp = int(match.group(1))
            freq_center = int(match.group(2))
            sample_rate = int(match.group(3))
            sample_type = match.group(4)
        except AttributeError:
            timestamp = None
            freq_center = None
            sample_rate = None
            sample_type = None

    sample_dtype, sample_type = SAMPLE_DTYPES.get(sample_type, (None, None))
    sample_bits = None
    sample_len = None
    if sample_dtype:
        if is_fft(filename):
            sample_dtype = np.float32
            sample_bits = 32
            sample_len = 4
        else:
            sample_dtype = np.dtype([("i", sample_dtype), ("q", sample_dtype)])
            sample_bits = sample_dtype[0].itemsize * 8
            sample_len = sample_dtype[0].itemsize * 2
    file_info = {
        "filename": filename,
        "freq_center": freq_center,
        "sample_rate": sample_rate,
        "sample_dtype": sample_dtype,
        "sample_len": sample_len,
        "sample_type": sample_type,
        "sample_bits": sample_bits,
        "nfft": nfft,
        "timestamp": timestamp,
    }
    return file_info


def read_samples(filename, sample_dtype, sample_bytes, seek_bytes=0, nfft=None, n=None):
    reader = get_reader(filename)

    # with reader(filename) as infile:
    #     infile.seek(int(seek_bytes))
    #     sample_buffer_all = infile.readall()

    with reader(filename) as infile:
        infile.seek(int(seek_bytes))
        # print(f"{n=}")
        # print(f"{nfft=}")
        # print(f"{sample_bytes=}")
        sample_buffer = infile.read(n * nfft * sample_bytes)

        # print(f"{len(sample_buffer)=}, {len(sample_buffer_all)=}")
        # print(len(sample_buffer) == len(sample_buffer_all))
        buffered_samples = int(len(sample_buffer) / sample_bytes)

        if buffered_samples == 0:
            print(filename)
            return None
        if buffered_samples / nfft != n:
            print("incomplete")
            # return None

        x1d = np.frombuffer(sample_buffer, dtype=sample_dtype, count=buffered_samples)
        return x1d["i"] + np.csingle(1j) * x1d["q"]


def argument_parser():
    parser = argparse.ArgumentParser(description="Waterfall plotter from scan data")
    parser.add_argument(
        "--sample_dir", type=str, help="Directory with sample zst files."
    )
    parser.add_argument(
        "--min_freq", type=float, help="Minimum frequency for plotting."
    )
    parser.add_argument(
        "--max_freq", type=float, help="Maximum frequency for plotting."
    )
    parser.add_argument(
        "--sampling_rate", default=100e6, type=float, help="Sampling rate."
    )
    parser.add_argument("--nfft", default=256, type=int, help="FFT length.")
    parser.add_argument(
        "--write_samples",
        default=2048,
        type=int,
        help="Number of samples written during scan.",
    )

    return parser


def main():
    # ARG PARSE PARAMETERS
    parser = argument_parser()
    args = parser.parse_args()
    sample_dir = args.sample_dir
    min_freq = args.min_freq
    max_freq = args.max_freq
    nfft = args.nfft
    n = args.write_samples
    sps = args.sampling_rate

    noverlap = nfft // 8
    db_min = -220
    db_max = -70

    cmap = plt.get_cmap("turbo")

    plot_min_freq = min_freq  # freq - (sps/2)
    plot_max_freq = max_freq  # freq + (sps/2)

    freq_resolution = sps / nfft
    max_idx = round((max_freq - min_freq) / freq_resolution)
    total_time = (nfft * n) / sps
    expected_time_bins = int((nfft * n) / (nfft - noverlap))
    X, Y = np.meshgrid(
        np.linspace(
            plot_min_freq,
            plot_max_freq,
            int((max_freq - min_freq) / freq_resolution + 1),
        ),
        np.linspace(0, total_time, expected_time_bins),
    )
    freq_bin_vals = X[0]
    # print(f"{freq_bin_vals.shape=}{freq_bin_vals=}")

    fig = plt.figure(figsize=(28, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    title = ax.text(0.5, 1.02, "", transform=ax.transAxes, va="center", ha="center")
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=db_min, vmax=db_max)
    cbar_ax = fig.add_axes([0.87, 0.250, 0.03, 0.5])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("dB", rotation=0)

    spec_data = np.empty(X.shape)
    spec_data.fill(np.nan)
    mesh = ax.pcolormesh(X, Y, spec_data, shading="nearest")

    plt.show(block=False)
    plt.pause(0.1)
    background = fig.canvas.copy_from_bbox(fig.bbox)
    ax.draw_artist(mesh)
    fig.canvas.blit(ax.bbox)

    processed_files = []
    while True:
        sample_files = [
            f
            for f in sorted(os.listdir(sample_dir))
            if f.startswith("sample") and f.endswith(".zst")
        ]

        needs_processing = []
        processing_batch = []
        for f in sample_files:
            f = os.path.join(sample_dir, f)
            # freq_center, sample_rate, sample_dtype, sample_bytes, sample_type, sample_bits, _, timestamp = parse_filename(f)
            file_info = parse_filename(f)
            freq_center = file_info["freq_center"]
            sample_rate = file_info["sample_rate"]
            if (
                (
                    ((freq_center + (sample_rate / 2)) >= min_freq)
                    and ((freq_center + (sample_rate / 2)) <= max_freq)
                )
                or (
                    ((freq_center - (sample_rate / 2)) >= min_freq)
                    and ((freq_center - (sample_rate / 2)) <= max_freq)
                )
            ) and f not in processed_files:
                if (
                    not processing_batch
                    or processing_batch[-1]["freq_center"] < freq_center
                ):
                    processing_batch.append(file_info)
                else:
                    needs_processing.append(processing_batch)
                    processing_batch = [file_info]

        # for n in needs_processing:
        #     print(f"{len(n)=}\n{n=}\n\n")

        # process files
        if needs_processing:
            # f = needs_processing[0][0]["filename"]
            # print(f"{f=}")
            spec_data.fill(np.nan)
            for file_info in needs_processing[0]:
                filename = file_info["filename"]
                sample_dtype = file_info["sample_dtype"]
                sample_bytes = file_info["sample_len"]
                freq_center = file_info["freq_center"]
                sample_rate = file_info["sample_rate"]
                sample_type = file_info["sample_type"]
                sample_bits = file_info["sample_bits"]
                timestamp = file_info["timestamp"]
                # filename, freq_center, sample_rate, sample_dtype, sample_bytes, sample_type, sample_bits, _, timestamp = parse_filename(f)

                samples = read_samples(
                    filename, sample_dtype, sample_bytes, seek_bytes=0, nfft=nfft, n=n
                )
                if samples is None:
                    continue
                freq_bins, t_bins, S = signal.spectrogram(
                    samples,
                    sample_rate,
                    window=signal.hann(int(nfft), sym=True),
                    nperseg=nfft,
                    noverlap=noverlap,
                    detrend="constant",
                    return_onesided=False,
                )
                freq_bins = np.fft.fftshift(freq_bins)
                # print(f"{freq_bins.shape=}{freq_center + freq_bins=}")

                idx = np.array(
                    [
                        round((item - min_freq) / freq_resolution)
                        for item in freq_bins + freq_center
                    ]
                ).astype(int)
                # print(f"{idx=}")
                # print(f"{np.where(idx>=0)}")
                # print(f"{(freq_bins+freq_center)[np.where(idx>=0)]=}")
                # print(f"{freq_bins.shape=}, {S.shape=}")

                S = np.fft.fftshift(S, axes=0)
                S = S.T
                S = 10 * np.log10(S)
                # S_norm = (S-np.min(S))/(np.max(S) - np.min(S))
                print(f"{np.min(S)=}, {np.max(S)=}")
                S_norm = (S - db_min) / (db_max - db_min)
                # plt.figure()
                # plt.pcolormesh(t_bins, freq_bins, S)
                # plt.show()
                # time.sleep(10)
                # print(f"{filename=}, {round((max_freq-min_freq)/freq_resolution)=}")
                # print(f"{np.flatnonzero(idx>=0)=}")
                # print(f"{idx[np.flatnonzero(idx>=0)]=}")
                spec_data[
                    : S_norm.shape[0],
                    idx[np.flatnonzero((idx >= 0) & (idx <= max_idx))],
                ] = S_norm[:, np.flatnonzero((idx >= 0) & (idx <= max_idx))]
                processed_files.append(filename)

            fig.canvas.restore_region(background)
            # spec_data = np.empty(X.shape)

            try:
                mesh.set_array(cmap(spec_data))
            except Exception as e:
                print(f"{e}")
                continue
            # print(f"{filename=}, {idx[np.flatnonzero(idx>=0)]=}\n")
            title.set_text(str(datetime.datetime.fromtimestamp(timestamp)))
            ax.draw_artist(title)
            ax.draw_artist(mesh)
            fig.canvas.blit(ax.title.axes.figure.bbox)
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()
            # time.sleep(3)

            # print(f"{freq_bins=}")
            # print(f"{freq_bins.shape=}")
            # print(f"{t_bins=}")
            # print(f"{t_bins.shape=}")

            # print(f"{samples.shape=}")

            # print(f"{processed_files=}")
        else:
            print("waiting")
            quit()


if __name__ == "__main__":
    main()
