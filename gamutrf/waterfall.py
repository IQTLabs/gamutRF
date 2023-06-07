import signal
import sys
import time

import argparse
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.signal import find_peaks

from gamutrf.zmqreceiver import ZmqReceiver, parse_scanners

matplotlib.use("GTK3Agg")


def draw_waterfall(mesh, fig, ax, data, cmap):
    mesh.set_array(cmap(data))
    ax.draw_artist(mesh)


def draw_title(ax, title, title_text):
    title_text["Time"] = str(datetime.datetime.now())
    title.set_text(str(title_text))
    ax.draw_artist(title)


def argument_parser():
    parser = argparse.ArgumentParser(description="waterfall plotter from scan data")
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
        "--n_detect", default=0, type=int, help="Number of detected signals to plot."
    )
    parser.add_argument(
        "--plot_snr", action="store_true", help="Plot SNR rather than power."
    )
    parser.add_argument(
        "--scanners",
        default="127.0.0.1:8001",
        type=str,
        help="Scanner endpoints to use.",
    )
    return parser


def main():
    # ARG PARSE PARAMETERS
    parser = argument_parser()
    args = parser.parse_args()
    min_freq = args.min_freq
    max_freq = args.max_freq
    plot_snr = args.plot_snr
    top_n = args.n_detect
    fft_len = args.nfft
    sampling_rate = args.sampling_rate

    # OTHER PARAMETERS
    cmap = plt.get_cmap("viridis")
    cmap_psd = plt.get_cmap("turbo")
    db_min = -220
    db_max = -150
    snr_min = 0
    snr_max = 50
    waterfall_height = 100  # number of waterfall rows
    scale = 1e6
    zmq_sleep_time = 1

    freq_resolution = sampling_rate / fft_len
    draw_rate = 1
    y_label_skip = 3
    psd_db_resolution = 90
    global init_fig
    init_fig = True
    points = [0]
    counter = 0
    y_ticks = []
    y_labels = []
    psd_x_edges = None
    psd_y_edges = None
    background = None
    top_n_lns = []

    fig = plt.figure(figsize=(28, 10), dpi=100)
    ax_psd: matplotlib.axes.Axes
    ax: matplotlib.axes.Axes
    mesh: matplotlib.collections.QuadMesh
    cbar_ax: matplotlib.axes.Axes
    cbar: matplotlib.colorbar.Colorbar
    sm: matplotlib.cm.ScalarMappable
    peak_lns: matplotlib.lines.Line2D
    current_psd_ln: matplotlib.lines.Line2D
    mean_psd_ln: matplotlib.lines.Line2D
    min_psd_ln: matplotlib.lines.Line2D
    max_psd_ln: matplotlib.lines.Line2D

    title_text = {}
    psd_title = ax_psd.text(
        0.5, 1.05, "", transform=ax_psd.transAxes, va="center", ha="center"
    )

    # SCALING
    min_freq /= scale
    max_freq /= scale
    freq_resolution /= scale
    scan_fres_resolution = 1e4

    # ZMQ
    zmqr = ZmqReceiver(
        scanners=parse_scanners(args.scanners),
        scan_fres=scan_fres_resolution,
    )

    def sig_handler(_sig=None, _frame=None):
        zmqr.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # PREPARE SPECTROGRAM
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

    def onresize(event):
        global init_fig
        init_fig = True

    fig.canvas.mpl_connect("resize_event", onresize)

    while True:
        if init_fig:
            # RESET FIGURE
            fig.clf()
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.15)
            ax_psd = fig.add_subplot(3, 1, 1)
            ax = fig.add_subplot(3, 1, (2, 3))

            # PSD
            XX, YY = np.meshgrid(
                np.linspace(
                    min_freq,
                    max_freq,
                    int((max_freq - min_freq) / (freq_resolution) + 1),
                ),
                np.linspace(db_min, db_max, psd_db_resolution),
            )

            psd_x_edges = XX[0]
            psd_y_edges = YY[:, 0]

            mesh_psd = ax_psd.pcolormesh(
                XX, YY, np.zeros(XX[:-1, :-1].shape), shading="flat"
            )
            (peak_lns,) = ax_psd.plot(
                X[0],
                db_min * np.ones(freq_data.shape[1]),
                color="white",
                marker="^",
                markersize=12,
                linestyle="none",
                fillstyle="full",
            )
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

            # SPECTROGRAM
            mesh = ax.pcolormesh(X, Y, db_data, shading="nearest")
            top_n_lns = []
            for _ in range(top_n):
                (ln,) = ax.plot(
                    [X[0][0]] * len(Y[:, 0]),
                    Y[:, 0],
                    color="brown",
                    linestyle=":",
                    alpha=0,
                )
                top_n_lns.append(ln)

            ax.set_xlabel("MHz")
            ax.set_ylabel("Time")

            # COLORBAR
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_clim(vmin=db_min, vmax=db_max)

            if plot_snr:
                sm.set_clim(vmin=snr_min, vmax=snr_max)
            cbar_ax = fig.add_axes([0.92, 0.10, 0.03, 0.5])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label("dB", rotation=0)

            # SPECTROGRAM TITLE
            title = ax.text(
                0.5, 1.05, "", transform=ax.transAxes, va="center", ha="center"
            )

            ax_psd.yaxis.set_animated(True)
            cbar_ax.yaxis.set_animated(True)
            ax.yaxis.set_animated(True)
            plt.show(block=False)
            plt.pause(0.1)

            background = fig.canvas.copy_from_bbox(fig.bbox)

            ax.draw_artist(mesh)
            fig.canvas.blit(ax.bbox)

            for ln in top_n_lns:
                ln.set_alpha(0.75)

            init_fig = False

        else:
            scan_configs, scan_df = zmqr.read_buff()

            if scan_df is not None:
                scan_config = scan_configs[0]
                scan_df = scan_df[(scan_df.freq > min_freq) & (scan_df.freq < max_freq)]
                if scan_df.empty:
                    print(
                        f"Scan is outside specified frequency range ({min_freq} to {max_freq})."
                    )
                    continue

                idx = (
                    round((scan_df.freq - min_freq) / freq_resolution)
                    .values.flatten()
                    .astype(int)
                )

                freq_data = np.roll(freq_data, -1, axis=0)
                freq_data[-1, :] = np.nan
                freq_data[-1][idx] = (
                    round(scan_df.freq / freq_resolution).values.flatten()
                    * freq_resolution
                )

                db = scan_df.db.values.flatten()

                db_data = np.roll(db_data, -1, axis=0)
                db_data[-1, :] = np.nan
                db_data[-1][idx] = db

                data, xedge, yedge = np.histogram2d(
                    freq_data[~np.isnan(freq_data)].flatten(),
                    db_data[~np.isnan(db_data)].flatten(),
                    density=False,
                    bins=[psd_x_edges, psd_y_edges],
                )
                heatmap = gaussian_filter(data, sigma=2)
                data = heatmap
                data /= np.max(data)
                # data /= np.max(data, axis=1)[:,None]

                fig.canvas.restore_region(background)

                top_n_bins = freq_bins[
                    np.argsort(np.nanvar(db_data - np.nanmin(db_data, axis=0), axis=0))[
                        ::-1
                    ][:top_n]
                ]

                for i, ln in enumerate(top_n_lns):
                    ln.set_xdata([top_n_bins[i]] * len(Y[:, 0]))

                fig.canvas.blit(ax.yaxis.axes.figure.bbox)

                row_time = datetime.datetime.fromtimestamp(scan_df.ts.iloc[-1])

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

                counter += 1
                if counter % draw_rate == 0:
                    draw_rate = 1

                    db_min = np.nanmin(db_data)
                    db_max = np.nanmax(db_data)

                    XX, YY = np.meshgrid(
                        np.linspace(
                            min_freq,
                            max_freq,
                            int((max_freq - min_freq) / (freq_resolution) + 1),
                        ),
                        np.linspace(db_min, db_max, psd_db_resolution),
                    )

                    psd_x_edges = XX[0]
                    psd_y_edges = YY[:, 0]

                    mesh_psd = ax_psd.pcolormesh(
                        XX, YY, np.zeros(XX[:-1, :-1].shape), shading="flat"
                    )

                    # db_norm = db_data
                    db_norm = (db_data - db_min) / (db_max - db_min)
                    if plot_snr:
                        db_norm = ((db_data - np.nanmin(db_data, axis=0)) - snr_min) / (
                            snr_max - snr_min
                        )

                    peaks, properties = find_peaks(
                        db_data[-1],
                        height=np.nanmean(db_data, axis=0),
                        width=3,
                        prominence=(0, 20),
                        rel_height=0.7,
                        wlen=120,
                    )

                    ax_psd.set_ylim(db_min, db_max)

                    mesh_psd.set_array(cmap_psd(data.T))
                    min_psd_ln.set_ydata(np.nanmin(db_data, axis=0))
                    max_psd_ln.set_ydata(np.nanmax(db_data, axis=0))
                    mean_psd_ln.set_ydata(np.nanmean(db_data, axis=0))
                    current_psd_ln.set_ydata(db_data[-1])
                    peak_lns.set_xdata(psd_x_edges[peaks])
                    peak_lns.set_ydata(properties["width_heights"])

                    ax_psd.draw_artist(mesh_psd)
                    if len(peaks) > 0:
                        vl = ax_psd.vlines(
                            x=psd_x_edges[peaks],
                            ymin=db_data[-1][peaks] - properties["prominences"],
                            ymax=db_data[-1][peaks],
                            color="white",
                        )
                        ax_psd.draw_artist(vl)
                        vl = ax_psd.vlines(
                            x=np.concatenate(
                                (
                                    psd_x_edges[properties["left_ips"].astype(int)],
                                    psd_x_edges[properties["right_ips"].astype(int)],
                                )
                            ),
                            ymin=db_min,
                            ymax=np.tile(db_data[-1][peaks], 2),
                            color="white",
                        )
                        ax_psd.draw_artist(vl)
                        for l_ips, r_ips, p in zip(
                            psd_x_edges[properties["left_ips"].astype(int)],
                            psd_x_edges[properties["right_ips"].astype(int)],
                            db_data[-1][peaks],
                        ):
                            shaded = ax_psd.fill_between(
                                [l_ips, r_ips], db_min, p, alpha=0.7
                            )
                            ax_psd.draw_artist(shaded)
                        hl = ax_psd.hlines(
                            y=properties["width_heights"],
                            xmin=psd_x_edges[properties["left_ips"].astype(int)],
                            xmax=psd_x_edges[properties["right_ips"].astype(int)],
                            color="white",
                        )
                        ax_psd.draw_artist(hl)
                        for l_ips, r_ips, p in zip(
                            psd_x_edges[properties["left_ips"].astype(int)],
                            psd_x_edges[properties["right_ips"].astype(int)],
                            peaks,
                        ):
                            txt = ax_psd.text(
                                l_ips + ((r_ips - l_ips) / 2),
                                (0.15 * (db_max - db_min)) + db_min,
                                f"f={l_ips + ((r_ips - l_ips)/2):.0f}MHz",
                                size=10,
                                ha="center",
                                color="white",
                                rotation=40,
                            )
                            ax_psd.draw_artist(txt)
                            txt = ax_psd.text(
                                l_ips + ((r_ips - l_ips) / 2),
                                (0.05 * (db_max - db_min)) + db_min,
                                f"BW={r_ips - l_ips:.0f}MHz",
                                size=10,
                                ha="center",
                                color="white",
                                rotation=40,
                            )
                            ax_psd.draw_artist(txt)

                    ax_psd.draw_artist(peak_lns)
                    ax_psd.draw_artist(min_psd_ln)
                    ax_psd.draw_artist(max_psd_ln)
                    ax_psd.draw_artist(mean_psd_ln)
                    ax_psd.draw_artist(current_psd_ln)

                    draw_waterfall(mesh, fig, ax, db_norm, cmap)
                    draw_title(ax_psd, psd_title, title_text)

                    sm.set_clim(vmin=db_min, vmax=db_max)
                    cbar.update_normal(sm)
                    cbar.draw_all()
                    cbar_ax.draw_artist(cbar_ax.yaxis)
                    fig.canvas.blit(cbar_ax.yaxis.axes.figure.bbox)
                    ax_psd.draw_artist(ax_psd.yaxis)
                    fig.canvas.blit(ax_psd.yaxis.axes.figure.bbox)
                    for ln in top_n_lns:
                        ax.draw_artist(ln)

                    ax.draw_artist(ax.yaxis)
                    fig.canvas.blit(ax_psd.bbox)
                    fig.canvas.blit(ax.yaxis.axes.figure.bbox)
                    fig.canvas.blit(ax.bbox)
                    fig.canvas.blit(cbar_ax.bbox)
                    fig.canvas.blit(fig.bbox)
                    fig.canvas.flush_events()

                    print(f"Plotting {row_time}")

                print("\n")

            else:
                print("Waiting for scanner (ZMQ)...")
                time.sleep(zmq_sleep_time)


if __name__ == "__main__":
    main()
