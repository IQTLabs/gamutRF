import argparse
import csv
import datetime
import json
import logging
import multiprocessing
import os
import shutil
import signal
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, current_app
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.ndimage import gaussian_filter

from gamutrf.peak_finder import get_peak_finder
from gamutrf.utils import SAMP_RATE, MIN_FREQ, MAX_FREQ, SCAN_FRES
from gamutrf.zmqreceiver import ZmqReceiver, parse_scanners

warnings.filterwarnings(action="ignore", message="Mean of empty slice")
warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
warnings.filterwarnings(action="ignore", message="Degrees of freedom <= 0 for slice.")


def safe_savefig(path):
    basename = os.path.basename(path)
    dirname = os.path.dirname(path)
    tmp_path = os.path.join(dirname, "." + basename)
    plt.savefig(tmp_path)
    os.rename(tmp_path, path)
    logging.info("wrote %s", path)
    return path


def draw_title(ax, title):
    title_text = {
        "Time": str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    }
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


def save_detections(
    config,
    state,
    scan_time,
    scan_configs,
    peaks,
    properties,
):
    detection_save_dir = Path(state.save_path, "detections")

    detection_config_save_path = str(
        Path(
            detection_save_dir,
            f"detections_scan_config_{scan_time}.json",
        )
    )
    detection_save_path = str(
        Path(
            detection_save_dir,
            f"detections_{scan_time}.csv",
        )
    )

    if not os.path.exists(detection_save_dir):
        os.makedirs(detection_save_dir)

    if state.previous_scan_config is None or state.previous_scan_config != scan_configs:
        state.previous_scan_config = scan_configs
        with open(detection_config_save_path, "w", encoding="utf8") as f:
            json.dump(
                {
                    "timestamp": scan_time,
                    "min_freq": config.min_freq,
                    "max_freq": config.max_freq,
                    "scan_configs": scan_configs,
                },
                f,
                indent=4,
            )

    if not os.path.exists(detection_save_path):
        with open(detection_save_path, "w", encoding="utf8") as detection_csv:
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

    with open(detection_save_path, "a", encoding="utf8") as detection_csv:
        writer = csv.writer(detection_csv)
        for i in range(len(peaks)):
            writer.writerow(
                [
                    scan_time,  # timestamp
                    state.psd_x_edges[
                        properties["left_ips"][i].astype(int)
                    ],  # start_freq
                    state.psd_x_edges[
                        properties["right_ips"][i].astype(int)
                    ],  # end_freq
                    properties["peak_heights"][i],  # dB
                    state.peak_finder.name,  # type
                ]
            )


def save_waterfall(
    config,
    state,
    save_time,
    scan_time,
    fig_path=None,
):
    now = datetime.datetime.now()
    if state.last_save_time is None:
        state.last_save_time = now

    if now - state.last_save_time > datetime.timedelta(minutes=save_time):
        waterfall_save_dir = Path(state.save_path, "waterfall")
        if not os.path.exists(waterfall_save_dir):
            os.makedirs(waterfall_save_dir)

        waterfall_save_path = str(
            Path(waterfall_save_dir, f"waterfall_{scan_time}.png")
        )
        if fig_path:
            shutil.copyfile(fig_path, waterfall_save_path)
        else:
            safe_savefig(waterfall_save_path)

        save_scan_configs = {
            "start_scan_timestamp": state.scan_times[0],
            "start_scan_config": state.scan_config_history[state.scan_times[0]],
            "end_scan_timestamp": state.scan_times[-1],
            "end_scan_config": state.scan_config_history[state.scan_times[-1]],
        }
        config_save_path = str(Path(waterfall_save_dir, f"config_{scan_time}.json"))
        with open(config_save_path, "w", encoding="utf8") as f:
            json.dump(save_scan_configs, f, indent=4)

        state.last_save_time = now
        logging.info(f"Saving {waterfall_save_path}")


def argument_parser():
    parser = argparse.ArgumentParser(description="waterfall plotter from scan data")
    parser.add_argument(
        "--min_freq", default=MIN_FREQ, type=float, help="Minimum frequency for plot."
    )
    parser.add_argument(
        "--max_freq", default=MAX_FREQ, type=float, help="Maximum frequency for plot."
    )
    parser.add_argument(
        "--sampling_rate", default=SAMP_RATE, type=float, help="Sampling rate."
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
    parser.add_argument(
        "--port",
        default=0,
        type=int,
        help="If set, serve waterfall on this port.",
    )
    parser.add_argument(
        "--rotate_secs",
        default=900,
        type=int,
        help="If > 0, rotate save directories every N seconds",
    )
    parser.add_argument(
        "--width",
        default=28,
        type=float,
        help="Waterfall image width",
    )
    parser.add_argument(
        "--height",
        default=10,
        type=float,
        help="Waterfall image height",
    )
    parser.add_argument(
        "--waterfall_height",
        default=100,
        type=int,
        help="Waterfall height",
    )
    return parser


def reset_mesh_psd(config, state, data=None):
    if state.mesh_psd:
        state.mesh_psd.remove()

    XX, YY = np.meshgrid(
        np.linspace(
            config.min_freq,
            config.max_freq,
            int((config.max_freq - config.min_freq) / (config.freq_resolution) + 1),
        ),
        np.linspace(state.db_min, state.db_max, config.psd_db_resolution),
    )

    state.psd_x_edges = XX[0]
    state.psd_y_edges = YY[:, 0]

    if data is None:
        data = np.zeros(XX[:-1, :-1].shape)

    state.mesh_psd = state.ax_psd.pcolormesh(XX, YY, data, shading="flat")


def reset_mesh(state, data):
    if state.mesh:
        state.mesh.remove()
    state.mesh = state.ax.pcolormesh(state.X, state.Y, data, shading="nearest")


def reset_fig(
    config,
    state,
):
    # RESET FIGURE
    state.fig.clf()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    state.ax_psd = state.fig.add_subplot(3, 1, 1)
    state.ax = state.fig.add_subplot(3, 1, (2, 3))
    state.psd_title = state.ax_psd.text(
        0.5, 1.05, "", transform=state.ax_psd.transAxes, va="center", ha="center"
    )

    reset_mesh_psd(config, state)
    (state.peak_lns,) = state.ax_psd.plot(
        state.X[0],
        state.db_min * np.ones(state.freq_data.shape[1]),
        color="white",
        marker="^",
        markersize=12,
        linestyle="none",
        fillstyle="full",
    )
    (state.max_psd_ln,) = state.ax_psd.plot(
        state.X[0],
        state.db_min * np.ones(state.freq_data.shape[1]),
        color="red",
        marker=",",
        linestyle=":",
        markevery=config.marker_distance,
        label="max",
    )
    (state.min_psd_ln,) = state.ax_psd.plot(
        state.X[0],
        state.db_min * np.ones(state.freq_data.shape[1]),
        color="pink",
        marker=",",
        linestyle=":",
        markevery=config.marker_distance,
        label="min",
    )
    (state.mean_psd_ln,) = state.ax_psd.plot(
        state.X[0],
        state.db_min * np.ones(state.freq_data.shape[1]),
        color="cyan",
        marker="^",
        markersize=8,
        fillstyle="none",
        linestyle=":",
        markevery=config.marker_distance,
        label="mean",
    )
    (state.current_psd_ln,) = state.ax_psd.plot(
        state.X[0],
        state.db_min * np.ones(state.freq_data.shape[1]),
        color="red",
        marker="o",
        markersize=8,
        fillstyle="none",
        linestyle=":",
        markevery=config.marker_distance,
        label="current",
    )
    state.ax_psd.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    state.ax_psd.set_ylabel("dB")

    # SPECTROGRAM
    reset_mesh(state, state.db_data)
    state.top_n_lns = []
    for _ in range(config.top_n):
        (ln,) = state.ax.plot(
            [state.X[0][0]] * len(state.Y[:, 0]),
            state.Y[:, 0],
            color="brown",
            linestyle=":",
            alpha=0,
        )
        state.top_n_lns.append(ln)

    state.ax.set_xlabel("MHz")
    state.ax.set_ylabel("Time")

    # COLORBAR
    state.sm = plt.cm.ScalarMappable(cmap=state.cmap)
    state.sm.set_clim(vmin=state.db_min, vmax=state.db_max)

    if config.plot_snr:
        state.sm.set_clim(vmin=config.snr_min, vmax=config.snr_max)
    state.cbar_ax = state.fig.add_axes([0.92, 0.10, 0.03, 0.5])
    state.cbar = state.fig.colorbar(state.sm, cax=state.cbar_ax)
    state.cbar.set_label("dB", rotation=0)

    # SPECTROGRAM TITLE
    _title = state.ax.text(
        0.5, 1.05, "", transform=state.ax.transAxes, va="center", ha="center"
    )

    for ax in (state.ax.xaxis, state.ax_psd.xaxis):
        ax.set_major_locator(MultipleLocator(state.major_tick_separator))
        ax.set_major_formatter("{x:.0f}")
        ax.set_minor_locator(state.minor_tick_separator)

    for ax in (state.ax_psd.yaxis, state.cbar_ax.yaxis, state.ax.yaxis):
        ax.set_animated(True)

    if not config.batch:
        plt.show(block=False)

    if not config.batch:
        state.background = state.fig.canvas.copy_from_bbox(state.fig.bbox)

    state.ax.draw_artist(state.mesh)
    state.fig.canvas.blit(state.ax.bbox)
    for ln in state.top_n_lns:
        ln.set_alpha(0.75)

    if not config.batch and config.savefig_path:
        safe_savefig(config.savefig_path)


def init_state(
    config,
    state,
):
    state.X, state.Y = np.meshgrid(
        np.linspace(
            config.min_freq,
            config.max_freq,
            int((config.max_freq - config.min_freq) / config.freq_resolution + 1),
        ),
        np.linspace(1, config.waterfall_height, config.waterfall_height),
    )

    state.freq_bins = state.X[0]
    state.db_data = np.empty(state.X.shape)
    state.db_data.fill(np.nan)
    state.freq_data = np.empty(state.X.shape)
    state.freq_data.fill(np.nan)


def init_fig(
    config,
    state,
    onresize,
):
    plt.close("all")

    state.cmap = plt.get_cmap("viridis")
    state.cmap_psd = plt.get_cmap("turbo")
    state.minor_tick_separator = AutoMinorLocator()
    n_ticks = min(
        ((config.max_freq - config.min_freq) / 100) * 5,
        20,
    )
    state.major_tick_separator = config.base * round(
        ((config.max_freq - config.min_freq) / n_ticks) / config.base
    )

    plt.rcParams["savefig.facecolor"] = "#2A3459"
    plt.rcParams["figure.facecolor"] = "#2A3459"
    for param in (
        "text.color",
        "axes.labelcolor",
        "xtick.color",
        "ytick.color",
        "axes.facecolor",
    ):
        plt.rcParams[param] = "#d2d5dd"

    state.fig = plt.figure(figsize=(config.width, config.height), dpi=100)
    if not config.batch:
        state.fig.canvas.mpl_connect("resize_event", onresize)


def draw_peaks(
    config,
    state,
    scan_time,
    scan_configs,
):
    peaks, properties = state.peak_finder.find_peaks(state.db_data[-1])
    peaks, properties = filter_peaks(peaks, properties)

    if state.save_path:
        save_detections(
            config,
            state,
            scan_time,
            scan_configs,
            peaks,
            properties,
        )

    state.peak_lns.set_xdata(state.psd_x_edges[peaks])
    state.peak_lns.set_ydata(properties["width_heights"])

    for child in state.ax_psd.get_children():
        if isinstance(child, LineCollection):
            child.remove()

    for i in range(len(state.detection_text) - 1, -1, -1):
        state.detection_text[i].set_visible(False)
        state.detection_text.pop(i)

    if len(peaks) > 0:
        # if False:
        vl_center = state.ax_psd.vlines(
            x=state.psd_x_edges[peaks],
            ymin=state.db_data[-1][peaks] - properties["prominences"],
            ymax=state.db_data[-1][peaks],
            color="white",
        )
        state.ax_psd.draw_artist(vl_center)
        vl_edges = state.ax_psd.vlines(
            x=np.concatenate(
                (
                    state.psd_x_edges[properties["left_ips"].astype(int)],
                    state.psd_x_edges[properties["right_ips"].astype(int)],
                )
            ),
            ymin=state.db_min,
            ymax=np.tile(state.db_data[-1][peaks], 2),
            color="white",
        )
        state.ax_psd.draw_artist(vl_edges)
        for l_ips, r_ips, p in zip(
            state.psd_x_edges[properties["left_ips"].astype(int)],
            state.psd_x_edges[properties["right_ips"].astype(int)],
            state.db_data[-1][peaks],
        ):
            shaded = state.ax_psd.fill_between(
                [l_ips, r_ips], state.db_min, p, alpha=0.7
            )
            state.ax_psd.draw_artist(shaded)
        hl = state.ax_psd.hlines(
            y=properties["width_heights"],
            xmin=state.psd_x_edges[properties["left_ips"].astype(int)],
            xmax=state.psd_x_edges[properties["right_ips"].astype(int)],
            color="white",
        )
        state.ax_psd.draw_artist(hl)
        for l_ips, r_ips, p in zip(
            state.psd_x_edges[properties["left_ips"].astype(int)],
            state.psd_x_edges[properties["right_ips"].astype(int)],
            peaks,
        ):
            for txt in (
                state.ax_psd.text(
                    l_ips + ((r_ips - l_ips) / 2),
                    (0.15 * (state.db_max - state.db_min)) + state.db_min,
                    f"f={l_ips + ((r_ips - l_ips)/2):.0f}MHz",
                    size=10,
                    ha="center",
                    color="white",
                    rotation=40,
                ),
                state.ax_psd.text(
                    l_ips + ((r_ips - l_ips) / 2),
                    (0.05 * (state.db_max - state.db_min)) + state.db_min,
                    f"BW={r_ips - l_ips:.0f}MHz",
                    size=10,
                    ha="center",
                    color="white",
                    rotation=40,
                ),
            ):
                state.detection_text.append(txt)
                state.ax_psd.draw_artist(txt)


def update_fig(config, state, zmqr, rotate_secs, save_time):
    if not state.fig or not state.ax:
        raise NotImplementedError

    results = []
    while True:
        scan_configs, scan_df = zmqr.read_buff()
        if scan_df is None:
            break
        results.append((scan_configs, scan_df))

    if not results:
        return False

    if config.base_save_path and rotate_secs:
        state.save_path = os.path.join(
            config.base_save_path,
            str(int(time.time() / rotate_secs) * rotate_secs),
        )
        if not os.path.exists(state.save_path):
            os.makedirs(state.save_path)

    for scan_configs, scan_df in results:
        scan_df = scan_df[
            (scan_df.freq >= config.min_freq) & (scan_df.freq <= config.max_freq)
        ]
        if scan_df.empty:
            logging.info(
                f"Scan is outside specified frequency range ({config.min_freq} to {config.max_freq})."
            )
            continue

        idx = (
            round((scan_df.freq - config.min_freq) / config.freq_resolution)
            .values.flatten()
            .astype(int)
        )

        state.freq_data = np.roll(state.freq_data, -1, axis=0)
        state.freq_data[-1, :] = np.nan
        state.freq_data[-1][idx] = (
            round(scan_df.freq / config.freq_resolution).values.flatten()
            * config.freq_resolution
        )

        db = scan_df.db.values.flatten()

        state.db_data = np.roll(state.db_data, -1, axis=0)
        state.db_data[-1, :] = np.nan
        state.db_data[-1][idx] = db
        state.db_min = np.nanmin(state.db_data)
        state.db_max = np.nanmax(state.db_data)

        state.data, _xedge, _yedge = np.histogram2d(
            state.freq_data[~np.isnan(state.freq_data)].flatten(),
            state.db_data[~np.isnan(state.db_data)].flatten(),
            density=False,
            bins=[state.psd_x_edges, state.psd_y_edges],
        )
        heatmap = gaussian_filter(state.data, sigma=2)
        state.data = heatmap
        state.data /= np.max(state.data)
        # data /= np.max(data, axis=1)[:,None]

        scan_time = scan_df.ts.iloc[-1]
        row_time = datetime.datetime.fromtimestamp(scan_time)
        state.scan_times.append(scan_time)
        state.scan_config_history[scan_time] = scan_configs
        if len(state.scan_times) > config.waterfall_height:
            remove_time = state.scan_times.pop(0)
            state.scan_config_history.pop(remove_time)

        if state.counter % config.y_label_skip == 0:
            state.y_labels.append(row_time.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            state.y_labels.append("")
        state.y_ticks.append(config.waterfall_height)
        for j in range(len(state.y_ticks) - 2, -1, -1):
            state.y_ticks[j] -= 1
            if state.y_ticks[j] < 1:
                state.y_ticks.pop(j)
                state.y_labels.pop(j)

        state.ax.set_yticks(state.y_ticks, labels=state.y_labels)

        state.counter += 1

        if state.counter % config.draw_rate == 0:
            if state.background:
                state.fig.canvas.restore_region(state.background)

            top_n_bins = state.freq_bins[
                np.argsort(
                    np.nanvar(state.db_data - np.nanmin(state.db_data, axis=0), axis=0)
                )[::-1][: config.top_n]
            ]

            for top_n_bin, ln in zip(top_n_bins, state.top_n_lns):
                ln.set_xdata([top_n_bin] * len(state.Y[:, 0]))

            state.fig.canvas.blit(state.ax.yaxis.axes.figure.bbox)

            reset_mesh_psd(config, state, data=state.cmap_psd(state.data.T))

            state.ax_psd.set_ylim(state.db_min, state.db_max)
            state.current_psd_ln.set_ydata(state.db_data[-1])

            state.min_psd_ln.set_ydata(np.nanmin(state.db_data, axis=0))
            state.max_psd_ln.set_ydata(np.nanmax(state.db_data, axis=0))
            state.mean_psd_ln.set_ydata(np.nanmean(state.db_data, axis=0))
            state.ax_psd.draw_artist(state.mesh_psd)

            if state.peak_finder:
                draw_peaks(
                    config,
                    state,
                    scan_time,
                    scan_configs,
                )

            for ln in (
                state.peak_lns,
                state.min_psd_ln,
                state.max_psd_ln,
                state.mean_psd_ln,
                state.current_psd_ln,
            ):
                state.ax_psd.draw_artist(ln)

            # db_norm = db_data
            db_norm = (state.db_data - state.db_min) / (state.db_max - state.db_min)
            if config.plot_snr:
                db_norm = (
                    (state.db_data - np.nanmin(state.db_data, axis=0)) - config.snr_min
                ) / (config.snr_max - config.snr_min)

            reset_mesh(state, state.cmap(db_norm))
            state.ax.draw_artist(state.mesh)

            draw_title(state.ax_psd, state.psd_title)

            state.sm.set_clim(vmin=state.db_min, vmax=state.db_max)
            state.cbar.update_normal(state.sm)
            for ax in (state.cbar_ax.yaxis, state.ax_psd.yaxis):
                state.cbar_ax.draw_artist(ax)
                state.fig.canvas.blit(ax.axes.figure.bbox)
            for ln in state.top_n_lns:
                state.ax.draw_artist(ln)

            state.ax.draw_artist(state.ax.yaxis)
            for bmap in (
                state.ax_psd.bbox,
                state.ax.yaxis.axes.figure.bbox,
                state.ax.bbox,
                state.cbar_ax.bbox,
                state.fig.bbox,
            ):
                state.fig.canvas.blit(bmap)
            state.fig.canvas.flush_events()
            fig_path = None
            if config.savefig_path:
                fig_path = safe_savefig(config.savefig_path)

            logging.info(f"Plotting {row_time}")

            if state.save_path:
                save_waterfall(
                    config,
                    state,
                    save_time,
                    scan_time,
                    fig_path=fig_path,
                )

    return True


class WaterfallConfig:
    def __init__(
        self,
        engine,
        plot_snr,
        savefig_path,
        sampling_rate,
        fft_len,
        min_freq,
        max_freq,
        top_n,
        base_save_path,
        width,
        height,
        waterfall_height,
        batch,
    ):
        self.engine = engine
        self.plot_snr = plot_snr
        self.savefig_path = savefig_path
        self.snr_min = 0
        self.snr_max = 50
        self.waterfall_height = waterfall_height  # number of waterfall rows
        self.marker_distance = 0.1
        self.scale = 1e6
        self.freq_resolution = sampling_rate / fft_len / self.scale
        self.psd_db_resolution = 90
        self.y_label_skip = 3
        self.base = 20
        self.min_freq = min_freq / self.scale
        self.max_freq = max_freq / self.scale
        self.top_n = top_n
        self.draw_rate = 1
        self.base_save_path = base_save_path
        self.width = width
        self.height = height
        self.batch = batch


class WaterfallState:
    def __init__(self):
        self.db_min = -220
        self.db_max = -150
        self.db_data = None
        self.freq_data = None
        self.freq_bins = None
        self.detection_text = []
        self.scan_times = []
        self.scan_config_history = {}
        self.y_ticks = []
        self.y_labels = []
        self.previous_scan_config = None
        self.last_save_time = None
        self.counter = 0
        self.minor_tick_separator = None
        self.major_tick_separator = None
        self.cmap_psd = None
        self.cmap = None
        self.X = None
        self.Y = None
        self.fig = None
        self.top_n_lns = None
        self.background = None
        self.mesh = None
        self.psd_title = None
        self.cbar_ax = None
        self.cbar = None
        self.psd_x_edges = None
        self.psd_y_edges = None
        self.min_psd_ln = None
        self.max_psd_ln = None
        self.mean_psd_ln = None
        self.current_psd_ln = None
        self.ax_psd = None
        self.ax = None
        self.save_path = None
        self.mesh_psd = None
        self.data = None
        self.peak_finder = None


def waterfall(
    min_freq,
    max_freq,
    plot_snr,
    top_n,
    fft_len,
    sampling_rate,
    base_save_path,
    save_time,
    peak_finder,
    engine,
    savefig_path,
    rotate_secs,
    width,
    height,
    waterfall_height,
    batch,
    zmqr,
):
    config = WaterfallConfig(
        engine,
        plot_snr,
        savefig_path,
        sampling_rate,
        fft_len,
        min_freq,
        max_freq,
        top_n,
        base_save_path,
        width,
        height,
        waterfall_height,
        batch,
    )
    state = WaterfallState()
    state.save_path = config.base_save_path
    init_state(config, state)
    state.peak_finder = peak_finder

    global need_reset_fig
    need_reset_fig = True
    global running
    running = True

    def onresize(_event):
        global need_reset_fig
        need_reset_fig = True

    def sig_handler(_sig=None, _frame=None):
        global running
        running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    matplotlib.use(config.engine)
    init_fig(config, state, onresize)

    while zmqr.healthy() and running:
        if need_reset_fig:
            reset_fig(config, state)
            need_reset_fig = False
        if not update_fig(config, state, zmqr, rotate_secs, save_time):
            time.sleep(0.1)
            continue
        if config.batch:
            init_fig(config, state, onresize)
            need_reset_fig = True

    zmqr.stop()


class FlaskHandler:
    def __init__(self, savefig_path, port, refresh=5):
        self.savefig_path = savefig_path
        self.refresh = refresh
        self.app = Flask(__name__, static_folder=os.path.dirname(savefig_path))
        self.savefig_file = os.path.basename(self.savefig_path)
        self.app.add_url_rule("/", "", self.serve)
        self.app.add_url_rule(
            "/" + self.savefig_file, self.savefig_file, self.serve_waterfall
        )
        self.process = multiprocessing.Process(
            target=self.app.run,
            kwargs={"host": "0.0.0.0", "port": port},  # nosec
        )

    def start(self):
        self.process.start()

    def serve(self):
        return (
            '<html><head><meta http-equiv="refresh" content="%u"></head><body><img src="%s"></img></body></html>'
            % (self.refresh, self.savefig_file),
            200,
        )

    def serve_waterfall(self):
        return current_app.send_static_file(self.savefig_file)


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")

    parser = argument_parser()
    args = parser.parse_args()
    detection_type = args.detection_type.lower()
    peak_finder = None

    if args.save_path:
        Path(args.save_path, "waterfall").mkdir(parents=True, exist_ok=True)

        if detection_type:
            Path(args.save_path, "detections").mkdir(parents=True, exist_ok=True)
            peak_finder = get_peak_finder(detection_type)

    with tempfile.TemporaryDirectory() as tempdir:
        flask = None
        savefig_path = None
        engine = "GTK3Agg"
        batch = False

        if args.port:
            engine = "agg"
            batch = True
            savefig_path = os.path.join(tempdir, "waterfall.png")
            flask = FlaskHandler(savefig_path, args.port)
            flask.start()

        zmqr = ZmqReceiver(
            scanners=parse_scanners(args.scanners),
            scan_fres=SCAN_FRES,
        )

        waterfall(
            args.min_freq,
            args.max_freq,
            args.plot_snr,
            args.n_detect,
            args.nfft,
            args.sampling_rate,
            args.save_path,
            args.save_time,
            peak_finder,
            engine,
            savefig_path,
            args.rotate_secs,
            args.width,
            args.height,
            args.waterfall_height,
            batch,
            zmqr,
        )


if __name__ == "__main__":
    main()
