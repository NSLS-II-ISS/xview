import importlib
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest


def test_dash_layout_and_callbacks_with_current_components(monkeypatch):
    pytest.importorskip("dash")
    pytest.importorskip("dash_bootstrap_components")
    # Replace only the site integration; exercise the real Dash app and server.
    xas = ModuleType("xas")
    tiled = ModuleType("xas.tiled_io")
    for name in ["filter_node_by_metadata_key", "filter_node_for_proposal",
                 "sort_nodes_by_metadata_key", "group_node_by_metadata_key",
                 "build_scan_tree_table"]:
        setattr(tiled, name, Mock())
    analysis = ModuleType("xas.analysis")
    analysis.check_scan = Mock()
    for name, module in [("xas", xas), ("xas.tiled_io", tiled), ("xas.analysis", analysis)]:
        monkeypatch.setitem(sys.modules, name, module)
    try:
        web = importlib.import_module("xview.xview_dash")
        client = web.app.server.test_client()
        for path in ["/", "/_dash-layout", "/_dash-dependencies"]:
            assert client.get(path).status_code == 200
    finally:
        sys.modules.pop("xview.xview_dash", None)
        sys.modules.pop("xview.dash_elements.app_components", None)
