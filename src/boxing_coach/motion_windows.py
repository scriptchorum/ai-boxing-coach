"""Time and visibility rules shared by the local motion review detectors."""
from statistics import median

ARM_INTERVAL_MS = 260.0
LEG_INTERVAL_MS = 450.0
MAX_SAMPLE_GAP_MS = 150.0
MIN_CONFIDENCE = 0.65


def continuous(times, left, right):
    return 0 < times[right] - times[left] <= MAX_SAMPLE_GAP_MS


def supported_signal(values, times):
    """A centred three-sample median, bridging only isolated 1-frame (<=80ms) dropouts."""
    result = [None] * len(values)
    for i in range(1, len(values) - 1):
        w = values[i - 1:i + 2]
        if None not in w and continuous(times, i - 1, i) and continuous(times, i, i + 1):
            result[i] = median(w)
        elif w.count(None) == 1 and times[i + 1] - times[i - 1] <= 80.0:
            valid = [v for v in w if v is not None]
            result[i] = (valid[0] + valid[1]) / 2.0
    return result


def excursions(signal, times, rest, prominence):
    """Yield observed departures and returns, with prominence above both endpoints."""
    start = None
    peak = None
    for i, value in enumerate(signal):
        if i and not continuous(times, i - 1, i):
            start = peak = None
        if value is None:
            start = peak = None
        elif value <= rest:
            if peak is not None and signal[peak] - max(signal[start], value) >= prominence:
                yield start, peak, i
            start, peak = i, None
        elif start is not None and (peak is None or value > signal[peak]):
            peak = i
