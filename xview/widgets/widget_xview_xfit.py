import os
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pkg_resources

from PyQt5 import  QtWidgets, QtCore, uic
from PyQt5.QtCore import QSettings, QThread
from PyQt5.QtWidgets import QMenu
from PyQt5.Qt import Qt
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
#     NavigationToolbar2QT as NavigationToolbar

from sys import platform
from pathlib import Path

from matplotlib.figure import Figure
# from xas.xasproject import XASDataSet
# from isstools.elements.figure_update import update_figure
# from isstools.dialogs.BasicDialogs import message_box
# from xas.file_io import load_binned_df_from_file, load_binned_df_and_extended_data_from_file
import copy
from xview.dialogs.FileMetadataDialog import FileMetadataDialog
from xview.xfit_classes.workers import Worker_Retrive_MatProj_Data as worker_matproj

if platform == 'darwin':
    ui_path = pkg_resources.resource_filename('xview', 'ui/ui_xfit.ui')
else:
    ui_path = pkg_resources.resource_filename('xview', 'ui/ui_xfit.ui')


class UIXFIT(*uic.loadUiType(ui_path)):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.parent = parent

        self.pushButton_search_matproj.clicked.connect(self.search_the_structure_from_materials_project)
        self.material_project_data = {}


    def search_materials_structure(self, formula='FeO'):
        print("Searching Materials...")

        self.thread_worker_matproj = QThread()
        self.worker_matproj = worker_matproj(formula=formula)
        self.worker_matproj.moveToThread(self.thread_worker_matproj)
        self.thread_worker_matproj.started.connect(self.worker_matproj.run)
        self.worker_matproj.finished.connect(self.thread_worker_matproj.quit)
        self.worker_matproj.finished.connect(self.populate_materials_structure)
        self.thread_worker_matproj.finished.connect(self.get_finished_status)
        self.worker_matproj.finished.connect(self.worker_matproj.deleteLater)
        self.thread_worker_matproj.start()


    def get_finished_status(self):
        print('Search complete')

    def populate_materials_structure(self):
        if self.worker_matproj.worker_document is not None:
            self.documents = self.worker_matproj.worker_document

        if len(self.documents) > 0:

            _labels = ['mp-ID', 'Formula', 'structure', 'E full(eV)']

            self.clear_treeWidget(tree_widget=self.treeWidget_structure, labels=_labels)
            _parent = self.treeWidget_strucure

            self._treeWidget = {}

            print("Loading Structure...\n")

            for key, doc in self.documents.items():
                _name_list = [doc.material_id.string,
                              doc.formula_pretty,
                              f"{doc.symmetry.crystal_system}",
                              f"{doc.energy_above_hull:2.3f}"]

                self._treeWidget[_name_list[0]] = self._make_item(parent=_parent, item_list=_name_list)

        else:
            print('No strucutre found')


    def clear_treeWidget(self, tree_widget=None, labels=None):
        tree_widget.clear()
        tree_widget.setHeaderLabels(labels)
        for i in range(len(labels)):
            tree_widget.setColumnWidth(i, 150)
        tree_widget.setSortingEnabled(True)


    def _make_item(self, parent=None, item_list=None):
        _item = QtWidgets.QTreeWidgetItem(parent, item_list)
        _item.setFlags(_item.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
        _item.setCheckState(0, Qt.Unchecked)
        return _item



    def search_the_structure_from_materials_project(self):
        _formula = self.lineEdit_formula.text()
        self.material_project_data[_formula] = self.search_materials_structure(_formula)