import importlib
import sys
from types import ModuleType
from unittest.mock import Mock

import pandas as pd
from PyQt5 import QtWidgets


def test_selection_plot_and_refresh_share_and_invalidate_preview_cache(monkeypatch, tmp_path):
    # Site packages are unavailable in CI; use real Qt, pandas and Matplotlib.
    reader = Mock(side_effect=lambda path: (pd.read_csv(path, sep=r"\s+"), "header"))
    integrations = {
        "xas.xasproject": {"XASDataSet": Mock()},
        "xas.file_io": {"load_binned_df_from_file": reader,
                        "load_binned_df_and_extended_data_from_file": Mock()},
        "isstools.elements.figure_update": {
            "update_figure": lambda axes, toolbar, canvas: [ax.clear() for ax in axes]},
        "isstools.dialogs.BasicDialogs": {"message_box": Mock()},
        "larch.io": {"read_ascii": Mock()},
    }
    for name, attributes in integrations.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = None
    try:
        module = importlib.import_module("xview.widgets.widget_xview_data")
        settings = Mock()
        settings.value.return_value = "/GPFS/xf08id/User Data"
        monkeypatch.setattr(module, "QSettings", Mock(return_value=settings))
        parent = Mock()
        widget = module.UIXviewData(parent=parent)
        (tmp_path / "scan.dat").write_text("energy i0 it iff\n1 2 1 4\n2 4 2 8\n")
        widget.working_folder = str(tmp_path)
        widget.get_file_list()
        widget.list_data.setCurrentRow(0)
        for row in range(widget.listWidget_data_numerator.count()):
            item = widget.listWidget_data_numerator.item(row)
            item.setSelected(item.text() in {"it", "iff"})
        widget.listWidget_data_denominator.setCurrentRow(0)
        widget.checkBox_ratio.setChecked(True)
        widget.checkBox_log_bin.setChecked(False)
        widget.checkBox_inv_bin.setChecked(False)
        widget.plot_xas_data()
        widget.plot_xas_data()
        assert reader.call_count == 1
        assert len(widget.figure_data.ax.lines) == 2
        assert [line.get_label() for line in widget.figure_data.ax.lines] == [
            "scan it/i0", "scan iff/i0",
        ]
        assert parent.set_figure.call_count == 2
        widget.get_file_list()  # Clearing selection must not dereference None.
        widget.list_data.setCurrentRow(0)
        assert reader.call_count == 2
    finally:
        if widget is not None:
            widget.close()
            widget.deleteLater()
        app.processEvents()
        sys.modules.pop("xview.widgets.widget_xview_data", None)
