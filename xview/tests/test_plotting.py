from itertools import product

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from xview.plotting import plot_scan


@pytest.mark.parametrize("ratio,logarithm,invert", list(product([False, True], repeat=3)))
def test_channel_values_labels_and_source_ownership(ratio, logarithm, invert):
    frame = pd.DataFrame({"energy": [10, 20], "i0": [2, 4], "it": [1, 2], "iff": [4, 8]})
    original = frame.copy(deep=True)
    ax = Figure().subplots()
    labels = plot_scan(ax, frame, "sample.001.dat", "energy", ["it", "iff"], "i0",
                       ratio=ratio, logarithm=logarithm, invert=invert)
    for channel, line, label in zip(["it", "iff"], ax.lines, labels):
        expected = frame[channel].to_numpy(copy=True)
        expected_label = channel
        if ratio:
            expected = expected / frame.i0.to_numpy()
            expected_label += "/i0"
        if logarithm:
            expected = np.log(expected)
            expected_label = f"ln ({expected_label})"
        if invert:
            expected = -expected
            expected_label = f"- {expected_label}"
        np.testing.assert_allclose(line.get_ydata(), expected)
        np.testing.assert_array_equal(line.get_xdata(), [10, 20])
        assert label == expected_label
        assert line.get_label() == f"sample.001 {expected_label}"
    pd.testing.assert_frame_equal(frame, original)


def test_missing_energy_uses_sample_index_and_raw_plot_needs_no_divisor():
    ax = Figure().subplots()
    plot_scan(ax, pd.DataFrame({"it": [1, 2]}), "scan.dat", "", ["it"], "missing")
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 1])
