import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

from xview.xfit_classes.workers import Worker_Retrive_MatProj_Data, Worker_Run_Feff_Calculation


def install_client(monkeypatch, search):
    client = ModuleType("mp_api.client")
    client.MPRester = MagicMock()
    client.MPRester.return_value.__enter__.return_value.materials.summary.search = search
    monkeypatch.setitem(sys.modules, "mp_api.client", client)
    return client.MPRester


def test_material_search_reuses_connection_and_deduplicates_formulas(monkeypatch):
    search = MagicMock(side_effect=[
        [SimpleNamespace(material_id="mp-1")], [SimpleNamespace(material_id="mp-2")],
    ])
    client = install_client(monkeypatch, search)
    worker = Worker_Retrive_MatProj_Data("FeO, NiO, FeO, ", api_key="test-key")
    finished, progress = [], []
    worker.finished.connect(lambda: finished.append(True))
    worker.progress.connect(progress.append)
    worker.run()
    client.assert_called_once_with("test-key")
    assert [call.kwargs for call in search.call_args_list] == [{"formula": "FeO"}, {"formula": "NiO"}]
    assert set(worker.worker_document) == {"mp-1", "mp-2"}
    assert finished == [True]
    assert progress == [50, 100]


def test_missing_api_key_reports_error_and_finishes(monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    monkeypatch.delenv("PMG_MAPI_KEY", raising=False)
    worker = Worker_Retrive_MatProj_Data()
    errors, finished = [], []
    worker.failed.connect(errors.append)
    worker.finished.connect(lambda: finished.append(True))
    worker.run()
    assert "MP_API_KEY" in errors[0]
    assert finished == [True]


def test_network_failure_discards_partial_results_and_finishes(monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "test-key")
    install_client(monkeypatch, MagicMock(side_effect=[
        [SimpleNamespace(material_id="mp-1")], ConnectionError("offline"),
    ]))
    worker = Worker_Retrive_MatProj_Data("FeO, NiO")
    errors, finished = [], []
    worker.failed.connect(errors.append)
    worker.finished.connect(lambda: finished.append(True))
    worker.run()
    assert errors == ["offline"]
    assert finished == [True]
    assert worker.worker_document == {}


def test_feff_failure_still_finishes(monkeypatch):
    xafs = ModuleType("larch.xafs")
    xafs.feffrunner = MagicMock(side_effect=OSError("missing input"))
    monkeypatch.setitem(sys.modules, "larch.xafs", xafs)
    worker = Worker_Run_Feff_Calculation({"mp-1": "/unused"})
    errors, finished = [], []
    worker.failed.connect(errors.append)
    worker.finished.connect(lambda: finished.append(True))
    worker.run()
    assert errors == ["missing input"]
    assert finished == [True]
