"""Scan plotting without GUI state or beamline dependencies."""

from pathlib import Path

import numpy as np


def plot_scan(ax, frame, filename, energy_key, numerators, denominator, *,
              ratio=False, logarithm=False, invert=False):
    """Plot selected channels and return their transformed axis labels."""
    labels = []
    divisor = frame[denominator].to_numpy(copy=False) if ratio else None
    for numerator in numerators:
        spectrum = frame[numerator].to_numpy(copy=False)
        label = numerator
        if ratio:
            spectrum = spectrum / divisor
            label = f"{numerator}/{denominator}"
        if logarithm:
            spectrum = np.log(spectrum)
            label = f"ln ({label})"
        if invert:
            spectrum = -spectrum
            label = f"- {label}"
        energy = frame[energy_key] if energy_key in frame else np.arange(spectrum.size)
        ax.plot(energy, spectrum, label=f"{Path(filename).stem} {label}")
        labels.append(label)
    return labels
