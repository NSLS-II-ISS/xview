"""Background workers for Materials Project searches and FEFF runs."""

import os

from PyQt5.QtCore import QObject, pyqtSignal


class Worker_Retrive_MatProj_Data(QObject):
    finished = pyqtSignal()
    failed = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, formula='FeO', api_key=None):
        super().__init__()
        self.formula_list = list(dict.fromkeys(
            value.strip() for value in formula.split(',') if value.strip()
        ))
        self.api_key = api_key
        self.worker_document = {}

    def run(self):
        self.worker_document = {}
        try:
            api_key = self.api_key or os.environ.get('MP_API_KEY') or os.environ.get('PMG_MAPI_KEY')
            if not api_key:
                raise ValueError('Set MP_API_KEY to your Materials Project API key before searching.')
            if not self.formula_list:
                return
            from mp_api.client import MPRester

            documents = {}
            with MPRester(api_key) as mpr:
                for index, formula in enumerate(self.formula_list, start=1):
                    for document in mpr.materials.summary.search(formula=formula):
                        documents[str(document.material_id)] = document
                    self.progress.emit(index * 100 // len(self.formula_list))
            self.worker_document = documents
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class Worker_Run_Feff_Calculation(QObject):
    finished = pyqtSignal()
    failed = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, feff_path):
        super().__init__()
        self.feff_path = feff_path

    def run(self):
        try:
            from larch.xafs import feffrunner

            for index, folder in enumerate(self.feff_path.values(), start=1):
                feffrunner(folder=folder, feffinp='feff.inp').run()
                self.progress.emit(index * 100 // len(self.feff_path))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()
