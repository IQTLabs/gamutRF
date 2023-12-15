import numpy as np
from scipy.signal import find_peaks


class PeakFinderBase:
    name = "base"

    def find_peaks(self, db_data, height=None):
        raise NotImplementedError


class PeakFinderNarrow:
    name = "narrowband"

    def find_peaks(self, db_data, height=None):
        if height is None:
            height = np.nanmean(db_data)
        peaks, properties = find_peaks(
            db_data,
            height=height,
            width=(1, 10),
            prominence=10,
            rel_height=0.7,
            wlen=120,
        )
        return peaks, properties


class PeakFinderWide:
    name = "wideband"

    def find_peaks(self, db_data, height=None):
        if height is None:
            height = np.nanmean(db_data) + 1
        peaks, properties = find_peaks(
            db_data,
            height=height,
            width=10,
            prominence=(0, 20),
            rel_height=0.7,
            wlen=120,
        )
        return peaks, properties


peak_finders = {
    "wideband": PeakFinderWide,
    "narrowband": PeakFinderNarrow,
    "wb": PeakFinderWide,
    "nb": PeakFinderNarrow,
}


def get_peak_finder(name):
    if not name:
        return None
    try:
        return peak_finders[name]()
    except KeyError:
        raise NotImplementedError
