import csv
import datetime
import json
import logging
import os
import shutil
import time
from pathlib import Path
import numpy as np
import matplotlib
from matplotlib import style
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.ndimage import gaussian_filter


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
        waterfall_width,
        batch,
        rotate_secs,
        save_time,
    ):
        self.engine = engine
        self.plot_snr = plot_snr
        self.savefig_path = savefig_path
        self.snr_min = 0
        self.snr_max = 50
        self.waterfall_height = waterfall_height  # number of waterfall rows
        self.marker_distance = 0.1
        self.scale = 1e6
        self.fft_len = fft_len
        self.sampling_rate = sampling_rate
        self.psd_db_resolution = 90
        self.y_label_skip = 3
        self.top_n = top_n
        self.draw_rate = 1
        self.base_save_path = base_save_path
        self.width = width
        self.height = height
        self.batch = batch
        self.reclose_interval = 25
        self.min_freq = min_freq / self.scale
        self.max_freq = max_freq / self.scale
        self.freq_range = self.max_freq - self.min_freq
        self.freq_resolution = self.sampling_rate / self.scale / self.fft_len * 2
        self.waterfall_width = int(
            min(self.freq_range / self.freq_resolution, waterfall_width)
        )
        self.freq_resolution = self.freq_range / self.waterfall_width
        self.n_ticks = 20
        self.rotate_secs = rotate_secs
        self.save_time = save_time

    def __eq__(self, other):
        for attr in ("fft_len", "sampling_rate", "min_freq", "max_freq"):
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True


class WaterfallState:
    def __init__(self, save_path, peak_finder, X, Y):
        self.X = X
        self.Y = Y
        self.db_min = -220
        self.db_max = -150
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
        self.save_path = save_path
        self.mesh_psd = None
        self.peak_finder = peak_finder
        self.last_plot = 0
        self.freq_bins = self.X[0]
        self.db_data = np.empty(self.X.shape)
        self.db_data.fill(np.nan)
        self.freq_data = np.empty(self.X.shape)
        self.freq_data.fill(np.nan)
        self.sm = None
        self.peak_lns = None


def make_config(
    scan_configs,
    min_freq,
    max_freq,
    engine,
    plot_snr,
    savefig_path,
    top_n,
    base_save_path,
    width,
    height,
    waterfall_height,
    waterfall_width,
    batch,
    rotate_secs,
    save_time,
):
    sampling_rate = max([scan_config["sample_rate"] for scan_config in scan_configs])
    fft_len = max([scan_config["nfft"] for scan_config in scan_configs])
    if min_freq == 0:
        min_freq = min([scan_config["freq_start"] for scan_config in scan_configs])
    if max_freq == 0:
        max_freq = max([scan_config["freq_end"] for scan_config in scan_configs])

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
        waterfall_width,
        batch,
        rotate_secs,
        save_time,
    )
    return config


class WaterfallPlot:
    def __init__(self, base_save_path, peak_finder, config, num):
        self.config = config
        self.num = num
        X, Y = self.meshgrid(1, config.waterfall_height, config.waterfall_height)
        self.state = WaterfallState(base_save_path, peak_finder, X, Y)
        matplotlib.use(self.config.engine)
        style.use("fast")

    def meshgrid(self, start, stop, num):
        return np.meshgrid(
            np.linspace(
                self.config.min_freq,
                self.config.max_freq,
                self.config.waterfall_width,
            ),
            np.linspace(start, stop, num),
        )

    def need_init(self):
        return (
            self.config.batch and self.state.counter % self.config.reclose_interval == 0
        )

    def init_fig(self, onresize):
        logging.info("initializing figure")

        self.state.cmap = plt.get_cmap("viridis")
        self.state.cmap_psd = plt.get_cmap("turbo")
        self.state.minor_tick_separator = AutoMinorLocator()
        self.state.major_tick_separator = MultipleLocator(
            self.config.freq_range / self.config.n_ticks
        )

        plt.rcParams["savefig.facecolor"] = "#1e1e1e"  # "#2A3459"
        plt.rcParams["figure.facecolor"] = "#1e1e1e"  # "#2A3459"
        for param in (
            "text.color",
            "axes.labelcolor",
            "xtick.color",
            "ytick.color",
            "axes.facecolor",
        ):
            plt.rcParams[param] = "#cdcdcd"  # "#d2d5dd"

        self.state.fig = plt.figure(
            num=self.num,
            clear=True,
            figsize=(self.config.width, self.config.height),
            dpi=100,
        )
        if not self.config.batch:
            self.state.fig.canvas.mpl_connect("resize_event", onresize)

    def close(self):
        if self.state.fig:
            plt.close(self.state.fig)
            self.state.fig = None

    def reset_mesh_psd(self, data=None):
        if self.state.mesh_psd:
            self.state.mesh_psd.remove()

        X, Y = self.meshgrid(
            self.state.db_min,
            self.state.db_max,
            self.config.psd_db_resolution,
        )
        self.state.psd_x_edges = X[0]
        self.state.psd_y_edges = Y[:, 0]

        if data is None:
            data = np.zeros(X[:-1, :-1].shape)

        self.state.mesh_psd = self.state.ax_psd.pcolormesh(X, Y, data, shading="flat")

    def reset_mesh(self, data):
        if self.state.mesh:
            self.state.mesh.remove()
        self.state.mesh = self.state.ax.pcolormesh(
            self.state.X, self.state.Y, data, shading="nearest"
        )

    def reset_fig(self):
        logging.info("resetting figure")

        self.state.fig.clf()
        self.state.fig.tight_layout()
        self.state.fig.subplots_adjust(hspace=0.15)
        self.state.fig.subplots_adjust(left=0.20)
        self.state.ax_psd = self.state.fig.add_subplot(3, 1, 1)
        self.state.ax = self.state.fig.add_subplot(3, 1, (2, 3))
        self.state.psd_title = self.state.ax_psd.text(
            0.5,
            1.05,
            "",
            transform=self.state.ax_psd.transAxes,
            va="center",
            ha="center",
        )
        default_data = self.state.db_min * np.ones(self.state.freq_data.shape[1])

        self.reset_mesh_psd()

        def ax_psd_plot(linestyle="--", **kwargs):
            return self.state.ax_psd.plot(
                self.state.X[0],
                default_data,
                markevery=int(len(self.state.X[0]) * self.config.marker_distance),
                linestyle=linestyle,
                linewidth=0.5,
                **kwargs,
            )

        if self.state.peak_finder:
            (self.state.peak_lns,) = ax_psd_plot(
                color="white",
                marker="^",
                markersize=12,
                linestyle="none",
                fillstyle="full",
            )
        (self.state.max_psd_ln,) = ax_psd_plot(
            color="red",
            # marker=",",
            label="max",
        )
        (self.state.min_psd_ln,) = ax_psd_plot(
            color="pink",
            # marker=",",
            label="min",
        )
        (self.state.mean_psd_ln,) = ax_psd_plot(
            color="cyan",
            marker="^",
            markersize=6,
            fillstyle="full",
            label="mean",
        )
        (self.state.current_psd_ln,) = ax_psd_plot(
            color="white",
            marker="o",
            markersize=6,
            fillstyle="full",
            label="current",
        )
        self.state.ax_psd.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        self.state.ax_psd.set_ylabel("dB")

        # SPECTROGRAM
        self.reset_mesh(self.state.db_data)
        self.state.top_n_lns = []
        for _ in range(self.config.top_n):
            (ln,) = self.state.ax.plot(
                [self.state.X[0][0]] * len(self.state.Y[:, 0]),
                self.state.Y[:, 0],
                color="brown",
                linestyle=":",
                alpha=0,
            )
            ln.set_alpha(0.75)
            self.state.top_n_lns.append(ln)

        self.state.ax.set_xlabel("MHz")
        self.state.ax.set_ylabel("Time")

        # COLORBAR
        self.state.sm = plt.cm.ScalarMappable(cmap=self.state.cmap)
        self.state.sm.set_clim(vmin=self.state.db_min, vmax=self.state.db_max)

        if self.config.plot_snr:
            self.state.sm.set_clim(vmin=self.config.snr_min, vmax=self.config.snr_max)
        self.state.cbar_ax = self.state.fig.add_axes([0.92, 0.10, 0.03, 0.5])
        self.state.cbar = self.state.fig.colorbar(self.state.sm, cax=self.state.cbar_ax)
        self.state.cbar.set_label("dB", rotation=0)

        # SPECTROGRAM TITLE
        _title = self.state.ax.text(
            0.5, 1.05, "", transform=self.state.ax.transAxes, va="center", ha="center"
        )

        for ax in (self.state.ax.xaxis, self.state.ax_psd.xaxis):
            ax.set_major_locator(self.state.major_tick_separator)
            if self.config.freq_resolution < 0.01:
                ax.set_major_formatter("{x:.1f}")
            else:
                ax.set_major_formatter("{x:.0f}")
            ax.set_minor_locator(self.state.minor_tick_separator)

        for ax in (
            self.state.ax_psd.yaxis,
            self.state.cbar_ax.yaxis,
            self.state.ax.yaxis,
        ):
            ax.set_animated(True)

        self.state.ax.draw_artist(self.state.mesh)
        self.state.fig.canvas.blit(self.state.ax.bbox)

        if not self.config.batch:
            self.state.fig.show(block=False)
            self.state.fig.canvas.flush_events()
            self.state.background = self.state.fig.canvas.copy_from_bbox(
                self.state.fig.bbox
            )
            if self.config.savefig_path:
                self.safe_savefig(self.config.savefig_path)

    def update_fig(self, results):
        if not self.state.fig or not self.state.ax:
            raise NotImplementedError

        if self.config.base_save_path and self.config.rotate_secs:
            self.state.save_path = os.path.join(
                self.config.base_save_path,
                str(
                    int(time.time() / self.config.rotate_secs) * self.config.rotate_secs
                ),
            )
            if not os.path.exists(self.state.save_path):
                Path(self.state.save_path).mkdir(parents=True, exist_ok=True)

        if len(results) > 1:
            logging.info("processing backlog of %u results", len(results))

        scan_duration = 0

        for scan_configs, scan_df in results:
            tune_step_hz = min(
                scan_config["tune_step_hz"] for scan_config in scan_configs
            )
            tune_step_fft = min(
                scan_config["tune_step_fft"] for scan_config in scan_configs
            )
            scan_duration = scan_df.ts.max() - scan_df.ts.min()
            tune_count = scan_df.tune_count.max()

            if scan_duration:
                tune_rate_hz = tune_count / scan_duration
                tune_dwell_ms = (scan_duration * 1e3) / tune_count
            else:
                tune_rate_hz = 0
                tune_dwell_ms = 0
            idx = (
                ((scan_df.freq - self.config.min_freq) / self.config.freq_resolution)
                .round()
                .clip(lower=0, upper=(self.state.db_data.shape[1] - 1))
                .values.flatten()
                .astype(int)
            )

            self.state.freq_data = np.roll(self.state.freq_data, -1, axis=0)
            self.state.freq_data[-1][idx] = scan_df.freq.values.flatten()

            self.state.db_data = np.roll(self.state.db_data, -1, axis=0)
            self.state.db_data[-1][idx] = scan_df.db.values.flatten()

            scan_time = scan_df.ts.iloc[-1]
            row_time = datetime.datetime.fromtimestamp(scan_time)
            if scan_time not in self.state.scan_config_history:
                self.state.scan_times.append(scan_time)
            self.state.scan_config_history[scan_time] = scan_configs
            while len(self.state.scan_times) > self.config.waterfall_height:
                remove_time = self.state.scan_times.pop(0)
                self.state.scan_config_history.pop(remove_time)

            if self.state.counter % self.config.y_label_skip == 0:
                self.state.y_labels.append(row_time.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                self.state.y_labels.append("")
            self.state.y_ticks.append(self.config.waterfall_height)
            for j in range(len(self.state.y_ticks) - 2, -1, -1):
                self.state.y_ticks[j] -= 1
                if self.state.y_ticks[j] < 1:
                    self.state.y_ticks.pop(j)
                    self.state.y_labels.pop(j)

            self.state.counter += 1

        if self.state.counter % self.config.draw_rate == 0:
            now = time.time()
            since_last_plot = 0
            if self.state.last_plot:
                since_last_plot = now - self.state.last_plot
            self.state.last_plot = now
            logging.info(
                f"Plotting {row_time} (seconds since last plot {since_last_plot})"
            )

            self.state.db_min = np.nanmin(self.state.db_data)
            self.state.db_max = np.nanmax(self.state.db_data)

            self.state.db_max += 0.10 * abs(self.state.db_max)
            if self.state.db_max - self.state.db_min < 20:
                self.state.db_max = self.state.db_min + 20

            data, _xedge, _yedge = np.histogram2d(
                self.state.freq_data[~np.isnan(self.state.freq_data)].flatten(),
                self.state.db_data[~np.isnan(self.state.db_data)].flatten(),
                density=False,
                bins=[self.state.psd_x_edges, self.state.psd_y_edges],
            )
            heatmap = gaussian_filter(data, sigma=2)
            data = heatmap / np.max(heatmap)

            db_norm = (self.state.db_data - self.state.db_min) / (
                self.state.db_max - self.state.db_min
            )
            if self.config.plot_snr:
                db_norm = (
                    (self.state.db_data - np.nanmin(self.state.db_data, axis=0))
                    - self.config.snr_min
                ) / (self.config.snr_max - self.config.snr_min)

            top_n_bins = self.state.freq_bins[
                np.argsort(
                    np.nanvar(
                        self.state.db_data - np.nanmin(self.state.db_data, axis=0),
                        axis=0,
                    )
                )[::-1][: self.config.top_n]
            ]

            self.state.ax.set_yticks(self.state.y_ticks, labels=self.state.y_labels)

            if self.state.background:
                self.state.fig.canvas.restore_region(self.state.background)

            for top_n_bin, ln in zip(top_n_bins, self.state.top_n_lns):
                ln.set_xdata([top_n_bin] * len(self.state.Y[:, 0]))

            self.state.fig.canvas.blit(self.state.ax.yaxis.axes.figure.bbox)

            self.reset_mesh_psd(data=self.state.cmap_psd(data.T))

            self.state.ax_psd.set_ylim(self.state.db_min, self.state.db_max)
            self.state.current_psd_ln.set_ydata(self.state.db_data[-1])
            for ln, ln_func in (
                (self.state.min_psd_ln, np.nanmin),
                (self.state.max_psd_ln, np.nanmax),
                (self.state.mean_psd_ln, np.nanmean),
            ):
                ln.set_ydata(ln_func(self.state.db_data, axis=0))
            self.state.ax_psd.draw_artist(self.state.mesh_psd)

            lns_to_draw = [
                self.state.min_psd_ln,
                self.state.max_psd_ln,
                self.state.mean_psd_ln,
                self.state.current_psd_ln,
            ]

            if self.state.peak_finder:
                self.draw_peaks(
                    scan_time,
                    scan_configs,
                )
                lns_to_draw.append(self.state.peak_lns)

            for ln in lns_to_draw:
                self.state.ax_psd.draw_artist(ln)

            self.reset_mesh(self.state.cmap(db_norm))
            self.state.ax.draw_artist(self.state.mesh)

            self.draw_title(
                self.state.ax_psd,
                self.state.psd_title,
                scan_duration,
                tune_step_hz,
                tune_step_fft,
                tune_rate_hz,
                tune_dwell_ms,
                self.config.sampling_rate,
                self.config.freq_resolution,
            )

            self.state.sm.set_clim(vmin=self.state.db_min, vmax=self.state.db_max)
            self.state.cbar.update_normal(self.state.sm)
            for ax in (self.state.cbar_ax.yaxis, self.state.ax_psd.yaxis):
                self.state.cbar_ax.draw_artist(ax)
                self.state.fig.canvas.blit(ax.axes.figure.bbox)
            for ln in self.state.top_n_lns:
                self.state.ax.draw_artist(ln)

            self.state.ax.draw_artist(self.state.ax.yaxis)
            for bmap in (
                self.state.ax_psd.bbox,
                self.state.ax.yaxis.axes.figure.bbox,
                self.state.ax.bbox,
                self.state.cbar_ax.bbox,
                self.state.fig.bbox,
            ):
                self.state.fig.canvas.blit(bmap)
            self.state.fig.canvas.flush_events()
            fig_path = None
            if self.config.savefig_path:
                fig_path = self.safe_savefig(self.config.savefig_path)

            if self.state.save_path:
                self.save_waterfall(
                    self.config.save_time,
                    scan_time,
                    fig_path=fig_path,
                )

    def draw_peaks(self, scan_time, scan_configs):
        peaks, properties = self.state.peak_finder.find_peaks(self.state.db_data[-1])
        peaks, properties = self.filter_peaks(peaks, properties)
        left_ips = properties["left_ips"].astype(int)
        right_ips = properties["right_ips"].astype(int)

        if self.state.save_path:
            self.save_detections(
                scan_time,
                scan_configs,
                peaks,
                properties,
            )

        if self.state.peak_finder:
            self.state.peak_lns.set_xdata(self.state.psd_x_edges[peaks])
            self.state.peak_lns.set_ydata(properties["width_heights"])

        for child in self.state.ax_psd.get_children():
            if isinstance(child, LineCollection):
                child.remove()

        for i in range(len(self.state.detection_text) - 1, -1, -1):
            self.state.detection_text[i].set_visible(False)
            self.state.detection_text.pop(i)

        if len(peaks) > 0:
            vl_center = self.state.ax_psd.vlines(
                x=self.state.psd_x_edges[peaks],
                ymin=self.state.db_data[-1][peaks] - properties["prominences"],
                ymax=self.state.db_data[-1][peaks],
                color="white",
            )
            self.state.ax_psd.draw_artist(vl_center)
            vl_edges = self.state.ax_psd.vlines(
                x=np.concatenate(
                    (
                        self.state.psd_x_edges[left_ips],
                        self.state.psd_x_edges[right_ips],
                    )
                ),
                ymin=self.state.db_min,
                ymax=np.tile(self.state.db_data[-1][peaks], 2),
                color="white",
            )
            self.state.ax_psd.draw_artist(vl_edges)
            for l_ips, r_ips, p in zip(
                self.state.psd_x_edges[left_ips],
                self.state.psd_x_edges[right_ips],
                self.state.db_data[-1][peaks],
            ):
                shaded = self.state.ax_psd.fill_between(
                    [l_ips, r_ips], self.state.db_min, p, alpha=0.7
                )
                self.state.ax_psd.draw_artist(shaded)
            hl = self.state.ax_psd.hlines(
                y=properties["width_heights"],
                xmin=self.state.psd_x_edges[left_ips],
                xmax=self.state.psd_x_edges[right_ips],
                color="white",
            )
            self.state.ax_psd.draw_artist(hl)
            for l_ips, r_ips, p in zip(
                self.state.psd_x_edges[left_ips],
                self.state.psd_x_edges[right_ips],
                peaks,
            ):
                for txt in (
                    self.state.ax_psd.text(
                        l_ips + ((r_ips - l_ips) / 2),
                        (0.15 * (self.state.db_max - self.state.db_min))
                        + self.state.db_min,
                        f"f={l_ips + ((r_ips - l_ips)/2):.0f}MHz",
                        size=10,
                        ha="center",
                        color="white",
                        rotation=40,
                    ),
                    self.state.ax_psd.text(
                        l_ips + ((r_ips - l_ips) / 2),
                        (0.05 * (self.state.db_max - self.state.db_min))
                        + self.state.db_min,
                        f"BW={r_ips - l_ips:.0f}MHz",
                        size=10,
                        ha="center",
                        color="white",
                        rotation=40,
                    ),
                ):
                    self.state.detection_text.append(txt)
                    self.state.ax_psd.draw_artist(txt)

    def draw_title(
        self,
        ax,
        title,
        scan_duration,
        tune_step_hz,
        tune_step_fft,
        tune_rate_hz,
        tune_dwell_ms,
        sample_rate,
        freq_resolution,
    ):
        title_text = {
            "Time": str(datetime.datetime.now().isoformat()),
            "Scan time": "%.2fs" % scan_duration,
            "Step FFTs": "%u" % tune_step_fft,
            "Step size": "%.2fMHz" % (tune_step_hz / 1e6),
            "Sample rate": "%.2fMsps" % (sample_rate / 1e6),
            "Resolution": "%.2fMHz" % freq_resolution,
            "Tune rate": "%.2fHz" % tune_rate_hz,
            "Tune dwell time": "%.2fms" % tune_dwell_ms,
        }
        title.set_fontsize(8)
        title.set_text(str(title_text))
        ax.draw_artist(title)

    def filter_peaks(self, peaks, properties):
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
        self,
        scan_time,
        scan_configs,
        peaks,
        properties,
    ):
        detection_save_dir = Path(self.state.save_path, "detections")

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
            Path(detection_save_dir).mkdir(parents=True, exist_ok=True)

        if (
            self.state.previous_scan_config is None
            or self.state.previous_scan_config != scan_configs
        ):
            self.state.previous_scan_config = scan_configs
            with open(detection_config_save_path, "w", encoding="utf8") as f:
                json.dump(
                    {
                        "timestamp": scan_time,
                        "min_freq": self.config.min_freq,
                        "max_freq": self.config.max_freq,
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
                        self.state.psd_x_edges[
                            properties["left_ips"][i].astype(int)
                        ],  # start_freq
                        self.state.psd_x_edges[
                            properties["right_ips"][i].astype(int)
                        ],  # end_freq
                        properties["peak_heights"][i],  # dB
                        self.state.peak_finder.name,  # type
                    ]
                )

    def save_waterfall(
        self,
        save_time,
        scan_time,
        fig_path=None,
    ):
        now = datetime.datetime.now()
        if self.state.last_save_time is None:
            self.state.last_save_time = now

        if now - self.state.last_save_time > datetime.timedelta(minutes=save_time):
            waterfall_save_dir = Path(self.state.save_path, "waterfall")
            if not os.path.exists(waterfall_save_dir):
                Path(waterfall_save_dir).mkdir(parents=True, exist_ok=True)

            waterfall_save_path = str(
                Path(waterfall_save_dir, f"waterfall_{scan_time}.png")
            )
            if fig_path:
                shutil.copyfile(fig_path, waterfall_save_path)
            else:
                self.safe_savefig(waterfall_save_path)

            save_scan_configs = {
                "start_scan_timestamp": self.state.scan_times[0],
                "start_scan_config": self.state.scan_config_history[
                    self.state.scan_times[0]
                ],
                "end_scan_timestamp": self.state.scan_times[-1],
                "end_scan_config": self.state.scan_config_history[
                    self.state.scan_times[-1]
                ],
            }
            config_save_path = str(Path(waterfall_save_dir, f"config_{scan_time}.json"))
            with open(config_save_path, "w", encoding="utf8") as f:
                json.dump(save_scan_configs, f, indent=4)

            self.state.last_save_time = now
            logging.info(f"Saving {waterfall_save_path}")

    def safe_savefig(self, path):
        basename = os.path.basename(path)
        dirname = os.path.dirname(path)
        tmp_path = os.path.join(dirname, "." + basename)
        self.state.fig.savefig(tmp_path)
        os.rename(tmp_path, path)
        logging.debug("wrote %s", path)
        return path


class WaterfallPlotManager:
    def __init__(self, base_save_path, peak_finder):
        self.plots = []
        self.config = None
        self.base_save_path = base_save_path
        self.peak_finder = peak_finder

    def config_changed(self, config):
        return self.config != config

    def add_plot(self, config, num):
        if not self.plots:
            self.config = config
        self.plots.append(
            WaterfallPlot(self.base_save_path, self.peak_finder, config, num)
        )

    def close(self):
        for plot in self.plots:
            plot.close()
        self.plots = []
        self.config = None

    def update_fig(self, results):
        for plot in self.plots:
            plot.update_fig(results)

    def reset_fig(self):
        for plot in self.plots:
            plot.reset_fig()

    def init_fig(self, onresize):
        for plot in self.plots:
            plot.init_fig(onresize)

    def need_init(self):
        if self.plots:
            return self.plots[0].need_init()
        return False
