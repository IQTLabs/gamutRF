import argparse
import csv
import datetime
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import signal
import sys
import time
import warnings

from matplotlib.artist import Artist
from matplotlib.collections import LineCollection
from matplotlib.text import Text
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

from gamutrf.zmqreceiver import ZmqReceiver, parse_scanners

matplotlib.use("GTK3Agg")
warnings.filterwarnings(action="ignore", message="Mean of empty slice")
warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
warnings.filterwarnings(action="ignore", message="Degrees of freedom <= 0 for slice.")


def draw_waterfall(mesh, fig, ax, data, cmap):
    mesh.set_array(cmap(data))
    ax.draw_artist(mesh)


def draw_title(ax, title, title_text):
    title_text["Time"] = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    title.set_text(str(title_text))
    ax.draw_artist(title)


def filter_peaks(peaks, properties):
    for i in range(len(peaks) - 1, -1, -1):  # start from end of list
        for j in range(len(peaks)):
            if i == j:
                continue
            if (properties["left_ips"][i] > properties["left_ips"][j]) and (
                properties["right_ips"][i] < properties["right_ips"][j]
            ):
                peaks = np.delete(peaks, i)
                for k in properties:
                    properties[k] = np.delete(properties[k], i)

                break
                # properties["left_ips"] = np.delete(properties["left_ips"], i)
                # properties["right_ips"] = np.delete(properties["right_ips"], i)
                # properties["width_heights"] = np.delete(properties["width_heights"], i)
    return peaks, properties


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
        "--detection_type",
        default="",
        type=str,
        help="Detection type to plot (wideband, narrowband).",
    )
    parser.add_argument(
        "--save_path", default="", type=str, help="Path to save screenshots."
    )
    parser.add_argument(
        "--save_time",
        default=1,
        type=int,
        help="Save screenshot every save_time minutes. Only used if save_path also defined.",
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
    save_path = args.save_path
    save_time = args.save_time
    detection_type = args.detection_type

    if save_path:
        Path(save_path, "waterfall").mkdir(parents=True, exist_ok=True)

    if save_path and detection_type:
        Path(save_path, "detections").mkdir(parents=True, exist_ok=True)

    detection_type = detection_type.lower()
    if detection_type:
        if detection_type in ["wb", "wide band", "wideband"]:
            detection_type = "wideband"
        elif detection_type in ["nb", "narrow band", "narrowband"]:
            detection_type = "narrowband"
        else:
            raise ValueError("detection_type must be 'narrowband' or 'wideband'")

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
    base = 20
    n_ticks = min(((max_freq / scale - min_freq / scale) / 100) * 5, 20)
    major_tick_separator = base * round(
        ((max_freq / scale - min_freq / scale) / n_ticks) / base
    )
    minor_tick_separator = AutoMinorLocator()

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
    last_save_time = None
    scan_config_history = {}
    scan_times = []
    vl_center = None
    vl_edges = None
    hl = None
    detection_text = []
    previous_scan_config = None
    detection_config_save_path = None
    detection_save_path = None

    plt.rcParams["savefig.facecolor"] = "#2A3459"
    plt.rcParams["figure.facecolor"] = "#2A3459"
    text_color = "#d2d5dd"
    plt.rcParams["text.color"] = text_color
    plt.rcParams["axes.labelcolor"] = text_color
    plt.rcParams["xtick.color"] = text_color
    plt.rcParams["ytick.color"] = text_color
    plt.rcParams["axes.facecolor"] = text_color

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
    psd_title: matplotlib.text.Text

    title_text = {}

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
    marker_distance = 0.1  # len(freq_bins)/100
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
            psd_title = ax_psd.text(
                0.5, 1.05, "", transform=ax_psd.transAxes, va="center", ha="center"
            )

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
                markevery=marker_distance,
                label="max",
            )
            (min_psd_ln,) = ax_psd.plot(
                X[0],
                db_min * np.ones(freq_data.shape[1]),
                color="pink",
                marker=",",
                linestyle=":",
                markevery=marker_distance,
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
                markevery=marker_distance,
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
                markevery=marker_distance,
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

            ax.xaxis.set_major_locator(MultipleLocator(major_tick_separator))
            ax.xaxis.set_major_formatter("{x:.0f}")
            ax.xaxis.set_minor_locator(minor_tick_separator)
            ax_psd.xaxis.set_major_locator(MultipleLocator(major_tick_separator))
            ax_psd.xaxis.set_major_formatter("{x:.0f}")
            ax_psd.xaxis.set_minor_locator(minor_tick_separator)

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
                scan_df = scan_df[
                    (scan_df.freq >= min_freq) & (scan_df.freq <= max_freq)
                ]
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

                scan_time = scan_df.ts.iloc[-1]
                scan_times.append(scan_time)
                if len(scan_times) > waterfall_height:
                    remove_time = scan_times.pop(0)
                    if save_path:
                        scan_config_history.pop(remove_time)
                        # assert len(scan_config_history) <= waterfall_height
                row_time = datetime.datetime.fromtimestamp(scan_time)

                if counter % y_label_skip == 0:
                    y_labels.append(row_time.strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    y_labels.append("")
                y_ticks.append(waterfall_height)
                for j in range(len(y_ticks) - 2, -1, -1):
                    y_ticks[j] -= 1
                    if y_ticks[j] < 1:
                        y_ticks.pop(j)
                        y_labels.pop(j)

                ax.set_yticks(y_ticks, labels=y_labels)

                if save_path:
                    scan_config_history[scan_time] = scan_configs

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

                    # ax_psd.clear()

                    ax_psd.set_ylim(db_min, db_max)
                    mesh_psd.set_array(cmap_psd(data.T))
                    current_psd_ln.set_ydata(db_data[-1])

                    min_psd_ln.set_ydata(np.nanmin(db_data, axis=0))
                    max_psd_ln.set_ydata(np.nanmax(db_data, axis=0))
                    mean_psd_ln.set_ydata(np.nanmean(db_data, axis=0))
                    ax_psd.draw_artist(mesh_psd)

                    if detection_type:
                        # NARROWBAND SIGNAL DETECT

                        if detection_type == "narrowband":
                            peaks, properties = find_peaks(
                                db_data[-1],
                                height=np.nanmean(db_data, axis=0),
                                width=(1, 10),
                                prominence=10,
                                rel_height=0.7,
                                wlen=120,
                            )

                        # WIDEBAND SIGNAL DETECT
                        elif detection_type == "wideband":
                            peaks, properties = find_peaks(
                                db_data[
                                    -1
                                ],  # db_data[-1] - np.nanmin(db_data, axis=0),#db_data[-1],
                                # height=np.nanmean(db_data, axis=0) - np.nanmin(db_data, axis=0),
                                height=np.nanmean(db_data, axis=0) + 1,
                                width=10,
                                prominence=(0, 20),
                                rel_height=0.7,
                                wlen=120,
                            )

                        peaks, properties = filter_peaks(peaks, properties)

                        if save_path:
                            if (
                                previous_scan_config is None
                                or previous_scan_config != scan_configs
                            ):
                                previous_scan_config = scan_configs
                                detection_config_save_path = str(
                                    Path(
                                        save_path,
                                        "detections",
                                        f"detections_scan_config_{scan_time}.json",
                                    )
                                )
                                with open(detection_config_save_path, "w") as f:
                                    json.dump(
                                        {
                                            "timestamp": scan_time,
                                            "min_freq": min_freq,
                                            "max_freq": max_freq,
                                            "scan_configs": scan_configs,
                                        },
                                        f,
                                        indent=4,
                                    )
                                detection_save_path = str(
                                    Path(
                                        save_path,
                                        "detections",
                                        f"detections_{scan_time}.csv",
                                    )
                                )
                                with open(detection_save_path, "w") as detection_csv:
                                    writer = csv.writer(detection_csv)
                                    writer.writerow(
                                        [
                                            "timestamp",
                                            "start_freq",
                                            "end_freq",
                                            "dB",
                                            "type",
                                        ]
                                    )

                            with open(detection_save_path, "a") as detection_csv:
                                writer = csv.writer(detection_csv)
                                for i in range(len(peaks)):
                                    writer.writerow(
                                        [
                                            scan_time,  # timestamp
                                            psd_x_edges[
                                                properties["left_ips"][i].astype(int)
                                            ],  # start_freq
                                            psd_x_edges[
                                                properties["right_ips"][i].astype(int)
                                            ],  # end_freq
                                            properties["peak_heights"][i],  # dB
                                            detection_type,  # type
                                        ]
                                    )

                        peak_lns.set_xdata(psd_x_edges[peaks])
                        peak_lns.set_ydata(properties["width_heights"])

                        for child in ax_psd.get_children():
                            if isinstance(child, LineCollection):
                                child.remove()

                        for i in range(len(detection_text) - 1, -1, -1):
                            detection_text[i].set_visible(False)
                            detection_text.pop(i)

                        if len(peaks) > 0:
                            # if False:
                            vl_center = ax_psd.vlines(
                                x=psd_x_edges[peaks],
                                ymin=db_data[-1][peaks] - properties["prominences"],
                                ymax=db_data[-1][peaks],
                                color="white",
                            )
                            ax_psd.draw_artist(vl_center)
                            vl_edges = ax_psd.vlines(
                                x=np.concatenate(
                                    (
                                        psd_x_edges[properties["left_ips"].astype(int)],
                                        psd_x_edges[
                                            properties["right_ips"].astype(int)
                                        ],
                                    )
                                ),
                                ymin=db_min,
                                ymax=np.tile(db_data[-1][peaks], 2),
                                color="white",
                            )
                            ax_psd.draw_artist(vl_edges)
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
                                detection_text.append(txt)
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
                                detection_text.append(txt)
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
                    # cbar.draw_all()
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

                    if save_path:
                        if last_save_time is None:
                            last_save_time = datetime.datetime.now()

                        if (
                            datetime.datetime.now() - last_save_time
                            > datetime.timedelta(minutes=save_time)
                        ):
                            waterfall_save_path = str(
                                Path(
                                    save_path, "waterfall", f"waterfall_{scan_time}.png"
                                )
                            )
                            fig.savefig(waterfall_save_path)

                            save_scan_configs = {
                                "start_scan_timestamp": scan_times[0],
                                "start_scan_config": scan_config_history[scan_times[0]],
                                "end_scan_timestamp": scan_times[-1],
                                "end_scan_config": scan_config_history[scan_times[-1]],
                            }
                            config_save_path = str(
                                Path(save_path, "waterfall", f"config_{scan_time}.json")
                            )
                            with open(config_save_path, "w") as f:
                                json.dump(save_scan_configs, f, indent=4)

                            last_save_time = datetime.datetime.now()
                            print(f"Saving {waterfall_save_path}")

                print("\n")

            else:
                print("Waiting for scanner (ZMQ)...")
                time.sleep(zmq_sleep_time)


if __name__ == "__main__":
    main()
