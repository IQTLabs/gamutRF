import argparse
import os
import matplotlib

matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, TextBox
import numpy as np
import time
from timeit import default_timer as timer
import datetime
from findpeaks import findpeaks
import signal
import subprocess
import sys
import shlex


sleep_time = 0.5


def read_log(log_file):
    while True:
        line = log_file.readline()
        if not line or not line.endswith("\n"):
            print("WAITING FOR SCANNER...\n")
            time.sleep(sleep_time)
            continue
        yield line


def draw_waterfall(mesh, fig, ax, data, cmap, background):
    start = timer()

    mesh.set_array(cmap(data))
    end = timer()
    print(f"Set mesh {end-start}")

    ax.draw_artist(mesh)
    # ax.draw_artist(ax.yaxis)
    # fig.canvas.blit(fig.bbox)
    # fig.canvas.flush_events()

    # start = timer()
    # fig.canvas.draw()
    # end = timer()
    # print(f"Draw {end-start}")
    # plt.pause(0.01)


def draw_title(ax, title, title_text):
    title_text["Time"] = str(datetime.datetime.now())
    # title.set_text(f"{datetime.datetime.now()} {data.shape=}")
    title.set_text(str(title_text))
    ax.draw_artist(title)


def argument_parser():
    parser = argparse.ArgumentParser(description="waterfall plotter from scan data")
    parser.add_argument(
        "--fft_log", default="fftlog.csv", type=str, help="Filepath for FFT log file."
    )
    parser.add_argument(
        "--min_freq", default=300e6, type=float, help="Minimum frequency for plot."
    )
    parser.add_argument(
        "--max_freq", default=6e9, type=float, help="Maximum frequency for plot."
    )
    parser.add_argument(
        "--sampling_rate", default=100e6, type=float, help="Sampling rate."
    )
    parser.add_argument("--nfft", default=256, type=int, help="FFT length.")
    parser.add_argument(
        "--n_detect", default=80, type=int, help="Number of detected signals to plot."
    )
    parser.add_argument(
        "--plot_snr", action="store_true", help="Plot SNR rather than power."
    )
    return parser


def main():
    # ARG PARSE PARAMETERS
    parser = argument_parser()
    args = parser.parse_args()
    min_freq = args.min_freq  # 5.7e9 #300e6
    max_freq = args.max_freq  # 5.9e9#6e9
    plot_snr = args.plot_snr
    top_n = args.n_detect
    fft_len = args.nfft
    sampling_rate = args.sampling_rate

    # OTHER PARAMETERS
    cmap = plt.get_cmap("turbo")
    db_min = -50
    db_max = 15
    snr_min = 0
    snr_max = 50
    waterfall_height = 100  # number of waterfall rows
    fft_log_scale = 1e6
    scale = 1e6

    freq_resolution = (
        sampling_rate / fft_len
    )  # 0.5e6 # freq_resolution = sampling_rate / fft_len
    draw_rate = 1
    y_label_skip = 3

    # SCALING
    min_freq /= scale
    max_freq /= scale
    freq_resolution /= scale

    # DATA
    X, Y = np.meshgrid(
        np.linspace(
            min_freq, max_freq, int((max_freq - min_freq) / freq_resolution + 1)
        ),
        np.linspace(1, waterfall_height, waterfall_height),
    )
    freq_bins = X[0]
    db_data = np.empty(X.shape)
    db_data.fill(np.nan)
    freq_data = np.empty(X.shape)
    freq_data.fill(np.nan)
    # print(f"{int((max_freq-min_freq)/freq_resolution+1)=}")
    # print(f"{freq_resolution=}")
    # print(f"{len(freq_bins)=},{freq_bins=}")

    # quit()

    # PLOTTING
    # fig, ax = plt.subplots(figsize=(24,10), dpi=100)
    fig = plt.figure(figsize=(28, 10), dpi=100)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    ax_psd = fig.add_subplot(3, 1, 1)
    ax = fig.add_subplot(3, 1, (2, 3))

    # psd_img = ax_psd.imshow(np.random.rand(100,freq_data.shape[1]), interpolation='nearest', origin='lower', cmap=cmap)
    psd_db_resolution = 90
    XX, YY = np.meshgrid(
        np.linspace(
            min_freq, max_freq, int((max_freq - min_freq) / (freq_resolution) + 1)
        ),
        np.linspace(db_min, db_max, psd_db_resolution),
    )

    psd_x_edges = XX[0]
    psd_y_edges = YY[:, 0]

    mesh_psd = ax_psd.pcolormesh(XX, YY, np.zeros(XX[:-1, :-1].shape), shading="flat")
    (max_psd_ln,) = ax_psd.plot(
        X[0],
        db_min * np.ones(freq_data.shape[1]),
        color="red",
        marker=",",
        linestyle=":",
        markevery=10,
        label="max",
    )
    (min_psd_ln,) = ax_psd.plot(
        X[0],
        db_min * np.ones(freq_data.shape[1]),
        color="pink",
        marker=",",
        linestyle=":",
        markevery=10,
        label="min",
    )
    (mean_psd_ln,) = ax_psd.plot(
        X[0],
        db_min * np.ones(freq_data.shape[1]),
        color="cyan",
        marker="^",
        markersize=8,
        fillstyle="none",
        linestyle=":",
        markevery=20,
        label="mean",
    )
    (var_psd_ln,) = ax_psd.plot(
        X[0],
        db_min * np.ones(freq_data.shape[1]),
        color="magenta",
        marker="s",
        markersize=8,
        fillstyle="none",
        linestyle=":",
        markevery=20,
        label="variance",
    )
    (current_psd_ln,) = ax_psd.plot(
        X[0],
        db_min * np.ones(freq_data.shape[1]),
        color="red",
        marker="o",
        markersize=8,
        fillstyle="none",
        linestyle=":",
        markevery=10,
        label="current",
    )
    ax_psd.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax_psd.set_ylabel("dB")

    mesh = ax.pcolormesh(X, Y, db_data, shading="nearest")
    # top_n_lns = ax.vlines(np.zeros(top_n), ymin=0, ymax=1, colors="teal")
    top_n_lns = []
    for _ in range(top_n):
        # ln, = ax.plot([X[0][0]]*len(Y[:,0]), Y[:,0], color="pink", marker="H", markersize=8, fillstyle="none", linestyle=":", markevery=10, alpha=0)

        (ln,) = ax.plot(
            [X[0][0]] * len(Y[:, 0]), Y[:, 0], color="brown", linestyle=":", alpha=0
        )
        top_n_lns.append(ln)
    ax.set_xlabel("MHz")
    ax.set_ylabel("Time")

    sm = plt.cm.ScalarMappable(cmap=cmap)

    sm.set_clim(vmin=db_min, vmax=db_max)
    if plot_snr:
        sm.set_clim(vmin=snr_min, vmax=snr_max)
    cbar_ax = fig.add_axes([0.92, 0.10, 0.03, 0.5])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("dB", rotation=0)
    title = ax.text(0.5, 1.05, "", transform=ax.transAxes, va="center", ha="center")
    psd_title = ax_psd.text(
        0.5, 1.05, "", transform=ax_psd.transAxes, va="center", ha="center"
    )
    title_text = {}
    y_ticks = []
    y_labels = []
    ax.yaxis.set_animated(True)
    plt.show(block=False)
    plt.pause(0.1)
    background = fig.canvas.copy_from_bbox(fig.bbox)
    background_psd = fig.canvas.copy_from_bbox(ax_psd.bbox)
    ax.draw_artist(mesh)
    # ax.draw_artist(top_n_lns)
    # ax.draw_artist(top_n_ln)
    fig.canvas.blit(ax.bbox)
    # textbox_ax = fig.add_axes([0.1, 0.05, 0.8, 0.075])
    # text_box = TextBox(textbox_ax, "Evaluate", textalignment="center")
    # def submit(val):
    #     print(f"\n\nNew value entered: {val}\n\n")
    # text_box.on_submit(submit)
    # slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
    # slider = RangeSlider(slider_ax, "dB threshold", db_min, db_max)
    # def slider_update(val):
    #     print(f"New dB threshold: {val[0]},{val[1]}")
    # slider.on_changed(slider_update)
    for ln in top_n_lns:
        ln.set_alpha(0.75)

    # waterfall = []
    waterfall_row = []
    freq_idx = 0
    timestamp_idx = 1
    db_idx = 2

    # Check if log exists and sleep if not
    if not os.path.exists(args.fft_log):
        print(f"Waiting for {args.fft_log}...")
        while not os.path.exists(args.fft_log):
            time.sleep(sleep_time)
        print(f"Found {args.fft_log}. Starting waterfall plot.")

    # Iterate log
    with open(args.fft_log, "r") as logfile:
        loglines = read_log(logfile)
        counter = 0
        start_read = timer()
        for line in loglines:
            start = timer()

            # Get line
            # expected format = ['frequency', 'timestamp', 'dB']
            split_line = line.split()

            # Convert to floats
            try:
                line_floats = [float(x) for x in split_line]
            except:
                continue

            # Skip if outside frequency range
            if line_floats[freq_idx] < min_freq or line_floats[freq_idx] > max_freq:
                print(f"{line_floats[freq_idx]=}, {min_freq=}, {max_freq=}")
                print("Outside freq range")
                continue

            # Append frequency bin or process time row
            if not waterfall_row or waterfall_row[-1][freq_idx] < line_floats[freq_idx]:
                waterfall_row.append(line_floats)
            else:
                end_read = timer()
                print(f"Read waterfall row {end_read-start_read} ")
                scan_time = end_read - start_read
                end = timer()
                # print(f"Read waterfall row {end-start}")
                start = timer()
                # waterfall.append(waterfall_row)

                # freq = np.array([round_half(item[freq_idx]) for item in waterfall_row])
                # idx = (1/freq_resolution)*(freq - min_freq)
                # idx = idx.astype(int)

                idx = np.array(
                    [
                        round((item[freq_idx] - min_freq) / freq_resolution)
                        for item in waterfall_row
                    ]
                ).astype(int)

                freq_data = np.roll(freq_data, -1, axis=0)
                freq_data[-1, :] = np.nan

                # freq_data[-1][idx] = [(1/freq_resolution)*(item[freq_idx] - min_freq) for item in waterfall_row]
                # freq_data[-1][idx] = [item[freq_idx] for item in waterfall_row]
                freq_data[-1][idx] = [
                    round(item[freq_idx] / freq_resolution) * freq_resolution
                    for item in waterfall_row
                ]

                # freq_idx = round((freq-min_freq) / freq_resolution)

                db = np.array([item[db_idx] for item in waterfall_row])
                # db = (db - db_min) / (db_max - db_min)
                db_data = np.roll(db_data, -1, axis=0)
                db_data[-1, :] = np.nan
                db_data[-1][idx] = db

                # a = freq_data[~np.isnan(freq_data)].flatten()
                # b = db_data[~np.isnan(db_data)].flatten()
                ##ax_psd.hist2d(freq_data[~np.isnan(freq_data)].flatten(), db_data[~np.isnan(db_data)].flatten(), bins=(freq_data.shape[0], 50))

                # print(f"{min_freq=}")
                # print(freq_data[~np.isnan(freq_data)].flatten())
                # print(db_data[~np.isnan(db_data)].flatten())
                data, xedge, yedge = np.histogram2d(
                    freq_data[~np.isnan(freq_data)].flatten(),
                    db_data[~np.isnan(db_data)].flatten(),
                    density=False,
                    bins=[psd_x_edges, psd_y_edges],
                )  # bins=(freq_data.shape[1], psd_db_resolution))#, range=[[0, 400], [db_min, db_max]])

                # print(f"{psd_x_edges.shape=}")
                # print(f"{psd_y_edges.shape=}")
                # print(f"{data.shape=}")
                # print(f"{xedge=}")
                # print(f"{yedge=}")

                data /= np.max(data)  # todo uncomment

                np.set_printoptions(threshold=sys.maxsize)
                # print(f"{data=}")
                fig.canvas.restore_region(background)

                # print(f"{data.shape=}")
                # print(np.sum(data))
                # psd_img.set_data()
                # print(f"{data.T.shape=}")
                # print(f"{X[0]=}")
                # print(f"{Y[:,0]=}")
                # print(f"{X[0].shape=}")
                # print(f"{Y[:,0].shape=}")
                # print(f"{mesh_psd._A.shape=}")
                mesh_psd.set_array(cmap(data.T))
                # mesh_psd.set_array(data.T)
                # print(f"{np.nanmin(db_data, axis=0)=}")

                top_n_bins = freq_bins[
                    np.argsort(np.nanvar(db_data - np.nanmin(db_data, axis=0), axis=0))[
                        ::-1
                    ][:top_n]
                ]
                # print(f"{top_n_bins=}")
                # top_n_lns.set_segments([[[bin, 0], [bin, 1]] for bin in top_n_bins])
                for i, ln in enumerate(top_n_lns):
                    ln.set_xdata([top_n_bins[i]] * len(Y[:, 0]))

                # print(f"found signals = {freq_bins[np.where(db_data[-1] - np.mean(np.nanmin(db_data, axis=0)) > 0)]}")
                # print(db_data[-1] - np.mean(np.nanmin(db_data, axis=0)))

                min_psd_ln.set_ydata(np.nanmin(db_data, axis=0))
                max_psd_ln.set_ydata(np.nanmax(db_data, axis=0))
                mean_psd_ln.set_ydata(np.nanmean(db_data, axis=0))
                var_psd_ln.set_ydata(np.nanvar(db_data, axis=0))
                current_psd_ln.set_ydata(db_data[-1])
                ax_psd.draw_artist(mesh_psd)

                ax_psd.draw_artist(min_psd_ln)
                # print(f"{np.nanmin(db_data, axis=0)=}")
                # print(f"{np.nanmin(db_data, axis=0).shape=}")
                ax_psd.draw_artist(max_psd_ln)
                ax_psd.draw_artist(mean_psd_ln)
                ax_psd.draw_artist(var_psd_ln)
                ax_psd.draw_artist(current_psd_ln)

                # psd_img.set_data(data.T)
                # print(np.sum(psd_img.get_array()))
                # print(f"{psd_img.get_array().shape=}")
                # print(f"{np.max(psd_img.get_array())=}")
                # ax_psd.draw_artist(psd_img)

                # draw_title(ax_psd, psd_title,{'testing':'testing'})

                # fig.canvas.blit(fig.bbox)
                fig.canvas.blit(ax.yaxis.axes.figure.bbox)
                fig.canvas.blit(ax_psd.bbox)
                # fig.canvas.flush_events()
                # fig.canvas.draw()
                # ax_psd.draw_artist(psd_hist)
                # print(f"{a.shape=}, {b.shape=}")
                # plt.hist2d(freq_data[~np.isnan(freq_data)].flatten(), db_data[~np.isnan(db_data)].flatten(), bins=(freq_data.shape[0], 50))
                # plt.show()
                # time.sleep(100)

                # fit_start = timer()
                # results = fp.fit(db_data)
                # fit_end = timer()
                # print(f"Fit {fit_end-fit_start}")

                row_time = datetime.datetime.fromtimestamp(
                    float(waterfall_row[-1][timestamp_idx])
                )

                if counter % y_label_skip == 0:
                    y_labels.append(row_time)
                else:
                    y_labels.append("")
                y_ticks.append(waterfall_height)
                for j in range(len(y_ticks) - 2, -1, -1):
                    y_ticks[j] -= 1
                    if y_ticks[j] < 1:
                        y_ticks.pop(j)
                        y_labels.pop(j)

                ax.set_yticks(y_ticks, labels=y_labels)
                end = timer()
                print(f"Process row {end-start}")

                counter += 1
                if counter % draw_rate == 0:
                    # if (end_read - start_read ) < 1:
                    #     draw_rate = 1
                    #     print(f"Draw rate = {draw_rate}")
                    # else:
                    #     draw_rate = 4
                    #     #draw_rate = int(end_read-start_read) + 1
                    #     print(f"Draw rate = {draw_rate}")
                    draw_rate = 2
                    print(f"Draw rate = {draw_rate}")
                    start = timer()
                    db_norm = (db_data - db_min) / (db_max - db_min)
                    if plot_snr:
                        db_norm = ((db_data - np.nanmin(db_data, axis=0)) - snr_min) / (
                            snr_max - snr_min
                        )
                    draw_waterfall(mesh, fig, ax, db_norm, cmap, background)
                    # title_text["Dim"] = db_data.shape
                    # title_text["Scan time"] = f"{scan_time:.2f}"

                    draw_title(ax_psd, psd_title, title_text)
                    # draw_title(ax, title, title_text)

                    for ln in top_n_lns:
                        ax.draw_artist(ln)
                    ax.draw_artist(ax.yaxis)
                    fig.canvas.blit(ax.yaxis.axes.figure.bbox)
                    fig.canvas.blit(ax.bbox)
                    fig.canvas.flush_events()
                    end = timer()
                    print(f"Redraw {end-start}")
                waterfall_row = [line_floats]
                start_read = timer()
                print("\n")


if __name__ == "__main__":
    main()
