from pymatgen.ext.matproj import MPRester
from PyQt5.QtCore import QObject, pyqtSignal


API_KEY = 'MqQSJqQj6Z923DT9I9eZCgvbEM9rbCxT' #iss user AIP gmail

class Worker_Retrive_MatProj_Data(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, formula='FeO'):
        super(QObject, self).__init__()
        self.formula = formula.split(',')
        self.worker_document = {}

        self.formula_list = [f.strip() for f in self.formula]


    def run(self):
        for _formula in self.formula_list:
            with MPRester(API_KEY) as mpr:
                _documents = mpr.summary.search(formula=_formula)

            for _d in _documents:
                _material_id = _d.material_id.string
                self.worker_document[_material_id] = _d

        self.finished.emit()