from pathlib import Path

import pytest
from PyQt5 import uic

from xview.resources import resource_filename


@pytest.mark.parametrize("name", [
    "ui/ui_xview.ui", "ui/ui_xview_data_new.ui", "ui/ui_xview_project.ui",
    "ui/ui_xview_stats.ui", "ui/ui_xview_rixs.ui", "ui/ui_xfit.ui",
    "dialogs/MetadataDialog.ui", "dialogs/FileMetadataDialog.ui",
])
def test_qt_can_load_packaged_forms(name):
    form, widget = uic.loadUiType(resource_filename("xview", name))
    assert form is not None
    assert widget is not None


@pytest.mark.parametrize("name", ["assets/style.css", "spectra_db/spectrum_start.json"])
def test_other_resources_are_available(name):
    assert Path(resource_filename("xview", name)).is_file()
