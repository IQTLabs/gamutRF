import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="waterfall plotter from scan data")
    parser.add_argument(
        "--min_freq",
        default=0,
        type=float,
        help="Minimum frequency for plot (or 0 for automatic).",
    )
    parser.add_argument(
        "--max_freq",
        default=0,
        type=float,
        help="Maximum frequency for plot (or 0 for automatic).",
    )
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
        help="Scanner FFT endpoints to use.",
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
    parser.add_argument(
        "--waterfall_width",
        default=5000,
        type=int,
        help="Waterfall width (maximum)",
    )
    parser.add_argument(
        "--refresh",
        default=5,
        type=int,
        help="Waterfall refresh time",
    )
    parser.add_argument(
        "--predictions",
        default=3,
        type=int,
        help="If set, render N recent predictions",
    )
    parser.add_argument(
        "--inference_server",
        default="",
        type=str,
        help="Address of scanner for inference feed",
    )
    parser.add_argument(
        "--inference_port",
        default=10002,
        type=int,
        help="Port on scanner to connect to inference feed",
    )
    parser.add_argument(
        "--api_endpoint",
        default="127.0.0.1:9001",
        type=str,
        help="Scanner API endpoints to use.",
    )
    return parser
