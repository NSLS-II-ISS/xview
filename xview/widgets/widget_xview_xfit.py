import os
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pkg_resources

from PyQt5 import  QtWidgets, QtCore, uic
from PyQt5.QtCore import QSettings, QThread
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMenu
from PyQt5.Qt import Qt
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
#     NavigationToolbar2QT as NavigationToolbar

from sys import platform
from pymatgen.io.feff.sets import MPXANESSet, MPEXAFSSet, FEFFDictSet
from xraydb import atomic_symbol
from larch.xafs.feffutils import get_feff_pathinfo
from xview.xfit_classes.lightshow_pymatgen_bug_fix import FEFFDictSet_modified
from xview.xfit_classes.plotting_tools import MplCanvas, create_pyqtgraph_widget
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from larch.xafs import xftf
from larch.xafs.feffdat import ff2chi, feffpath
from larch.symboltable import Group
from pathlib import Path

from matplotlib.figure import Figure
# from xas.xasproject import XASDataSet
# from isstools.elements.figure_update import update_figure
# from isstools.dialogs.BasicDialogs import message_box
# from xas.file_io import load_binned_df_from_file, load_binned_df_and_extended_data_from_file
import copy
from xview.dialogs.FileMetadataDialog import FileMetadataDialog
from xview.xfit_classes.workers import Worker_Retrive_MatProj_Data as worker_matproj
from xview.xfit_classes.workers import Worker_Run_Feff_Calculation as worker_feff
from xview.xfit_classes.utils import read_lineEdit_and_perform_sanity_check, get_default_feff_parameters

if platform == 'darwin':
    ui_path = pkg_resources.resource_filename('xview', 'ui/ui_xfit.ui')
else:
    ui_path = pkg_resources.resource_filename('xview', 'ui/ui_xfit.ui')


ATOMIC_SYMBOL_DICT = {'element': [atomic_symbol(i) for i in range(20, 93)]}
EDGES = {'edges': ['K', 'L3', 'L2', 'L1', 'M5']}

STRUCTURE_FOLDER = "/nsls2/data/iss/legacy/Sandbox/structure_data/"


class UIXFIT(*uic.loadUiType(ui_path)):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.parent = parent

        self.pushButton_search_matproj.clicked.connect(self.search_the_structure_from_materials_project)
        self.pushButton_run_feff.clicked.connect(self.perform_feff_calculation)
        self.pushButton_list_path.clicked.connect(self.populate_scattering_paths)
        self.material_project_data = {}
        self.structure_found = False
        self.path_added = False
        self.treeWidget_paths_dict = {}
        self.create_plot_item_for_simulated_chi()
        self.create_plot_item_for_simulated_ft()
        self.pushButton_add_scattering_path.clicked.connect(self.plot_scattering_path)

        windows = ['sine', 'hanning', 'parzen', 'welch', 'gaussian', 'kaiser']

        for key in ['_sim']:
            getattr(self, 'comboBox_window' + key).addItems(windows)

    #################################################################################################
    ## This section of code make search from the materials project database and populates the entires


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

            _labels = ['mp-ID', 'Formula', 'structure', 'E full(eV)', "Experimentally Observed"]
            self.clear_treeWidget(tree_widget=self.treeWidget_structure, labels=_labels)
            _parent = self.treeWidget_structure
            self._treeWidget = {}
            print("Loading Structure...\n")
            for key, doc in self.documents.items():
                _name_list = [doc.material_id.string,
                              doc.formula_pretty,
                              f"{doc.symmetry.crystal_system}",
                              f"{doc.energy_above_hull:2.3f}",
                              f"{not doc.theoretical}"]
                _status, _status_experimentally_observed = self.check_if_feff_files_will_be_good(doc)  # Must be removed when pymatgen update the code
                self._treeWidget[doc.material_id.string] = self._make_item(parent=_parent, item_list=_name_list)
                if _status:
                    self._treeWidget[doc.material_id.string].setBackground(0, QColor('yellow'))
                if _status_experimentally_observed:
                    self._treeWidget[doc.material_id.string].setBackground(4, QColor('lime'))
            self.structure_found = True
        else:
            print('No strucutre found')

    def check_if_feff_files_will_be_good(self, document):
        for el, amt in document.composition.items():
            if amt == 1:
                status = False
            else:
                status = True
        return status, not document.theoretical


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

    ## This section of code make search from the materials project database and populates the entires
    #################################################################################################

    #################################################################################################
    ## This section of code perform feff calculation on selected entry and populates the feffpaths

    def make_feff_folder(self, absorbing_atom_valid_input, edge_valid_input, absorbing_atom, edge):
        task_finished = False
        if absorbing_atom_valid_input and edge_valid_input:
            self.treeWidget_paths_dict['paths'] = {}
            for key in self._treeWidget.keys():
                if self._treeWidget[key].checkState(0) == 2:
                    self.treeWidget_paths_dict['paths'][key] = os.path.join(STRUCTURE_FOLDER,
                                                                            self.documents[key].formula_pretty, key)
                    default_feff_parameters = get_default_feff_parameters(title=self.documents[key].material_id.string)
                    feff_dict = FEFFDictSet_modified(absorbing_atom=absorbing_atom,
                                            structure=self.documents[key].structure,
                                            spectrum="EXAFS",
                                            config_dict=default_feff_parameters,
                                            edge=edge,
                                            radius=7.5)
                    feff_dict.write_input(self.treeWidget_paths_dict['paths'][key])
                    task_finished = True
        if not task_finished:
            print("Select at least one Structure!")


    def run_feff(self, path_dict):
        print("Searching Material...\n")

        self.thread_feff_runner = QThread()
        self.worker_feff = worker_feff(feff_path=path_dict)
        self.worker_feff.moveToThread(self.thread_feff_runner)
        self.thread_feff_runner.started.connect(self.worker_feff.run)
        self.worker_feff.finished.connect(self.thread_feff_runner.quit)
        # self.worker_feff.finished.connect(self.populate_materials_structure)
        # self.thread_feff_runner.finished.connect(self.get_finished_status)
        self.worker_feff.finished.connect(self.worker_feff.deleteLater)

        self.thread_feff_runner.start()


    def perform_feff_calculation(self):
        absorbing_atom, absorbing_atom_valid_input = read_lineEdit_and_perform_sanity_check(
            lineEdit=self.lineEdit_absorbing_atom,
            reference_dictionary=ATOMIC_SYMBOL_DICT,
            message="Provide Valid Absorbing atom!")
        edge, edge_valid_input = read_lineEdit_and_perform_sanity_check(lineEdit=self.lineEdit_edge,
                                                                        reference_dictionary=EDGES,
                                                                        message="Provide Valid Edge!")

        self.make_feff_folder(absorbing_atom_valid_input=absorbing_atom_valid_input,
                              edge_valid_input=edge_valid_input,
                              absorbing_atom=absorbing_atom,
                              edge=edge)

        self.run_feff(self.treeWidget_paths_dict['paths'])


    def populate_scattering_paths(self):

        _labels = ['ID', 'Feff File', 'R', 'Coordination Number', 'Weight', 'Legs', 'Geometry']
        self.clear_treeWidget(tree_widget=self.treeWidget_scattering_path, labels=_labels)
        _parent = self.treeWidget_scattering_path

        self.treeWidget_paths_dict['widgets'] = {}
        self.treeWidget_paths_dict['feff_info'] = {}

        for key in self.treeWidget_paths_dict['paths'].keys():
            self.treeWidget_paths_dict['feff_info'][key] = get_feff_pathinfo(self.treeWidget_paths_dict['paths'][key])

            self.treeWidget_paths_dict['widgets'][key] = {}
            for path in self.treeWidget_paths_dict['feff_info'][key].paths:
                _name_list = [f"{key}",
                              f"{path.filename}",
                              f"{path.reff:1.3f}",
                              f"{path.degen:.0f}",
                              f"{path.cwratio:3.2f}",
                              f"{path.nleg}",
                              f"{path.geom}"]

                self.treeWidget_paths_dict['widgets'][key][path.filename] = self._make_item(parent=_parent,
                                                                                            item_list=_name_list)
        self.path_added = True


        ## This section of code perform feff calculation on selected entry and populates the feffpaths
        #################################################################################################

        ##################################################################################################
        ###################### plotting tools #####################################


    def create_plot_item_for_simulated_chi(self, number_of_references=1):

        self.canvas_sim_chi = MplCanvas(parent=self)
        _navi = NavigationToolbar2QT(self.canvas_sim_chi, parent=self)
        self.verticalLayout_sim_chi.addWidget(_navi)
        self.verticalLayout_sim_chi.addWidget(self.canvas_sim_chi)

    def create_plot_item_for_simulated_ft(self):

        self.canvas_sim_ft = MplCanvas(parent=self)
        _navi = NavigationToolbar2QT(self.canvas_sim_ft, parent=self)
        self.verticalLayout_sim_ft.addWidget(_navi)
        self.verticalLayout_sim_ft.addWidget(self.canvas_sim_ft)

    #################################################################################################################
    ### This section of code plot the selected paths

    def create_chi_and_ft(self):
        self.sim_chi_ft = {}
        for dic in self.feff_paths.keys():
            for key in self.feff_paths[dic].keys():
                self.sim_chi_ft[dic][key] = ff2chi(self.feff_paths[dic][key])

            self.sim_chi_ft[dic]['summation'] = ff2chi(self.feff_paths[dic])

        _feff = []
        for key in self.feff_paths.keys():
            _feff.append(self.feff_paths[key].values())
        self.sim_chi_ft['summation'] = ff2chi(_feff)

    def plot_scattering_path(self):

        self.canvas_sim_chi.axes.cla()
        self.canvas_sim_ft.axes.cla()

        self.treeWidget_paths_dict['selected_feff_data'] = {}
        __kweight = self.spinBox_kweight_sim.value()
        __window = self.comboBox_window_sim.currentText()
        __kmin = self.doubleSpinBox_kmin_sim.value()
        __kmax = self.doubleSpinBox_kmax_sim.value()

        __sum_chi = 0
        __sum_k = 0
        count = 0
        for key in self.treeWidget_paths_dict['widgets'].keys():
            self.treeWidget_paths_dict['selected_feff_data'][key] = {}
            for k, value in self.treeWidget_paths_dict['widgets'][key].items():
                if value.checkState(0) == 2:
                    __path = os.path.join(self.treeWidget_paths_dict['paths'][key], k)
                    _feffpath = feffpath(filename=__path)
                    _group = ff2chi([_feffpath])
                    __sum_k += _group.k
                    __sum_chi += _group.chi
                    count += 1
                    xftf(_group.k, _group.chi, kweight=__kweight, window=__window, kmin=__kmin, kmax=__kmax,
                         group=_group)
                    self.treeWidget_paths_dict['selected_feff_data'][key][k] = [_feffpath, _group]
                    self.canvas_sim_chi.axes.plot(_group.k, _group.chi * _group.k ** __kweight, label=f"{key} {k}")
                    self.canvas_sim_ft.axes.plot(_group.r, _group.chir_mag, label=f"{key} {k}")

        __sum = Group()
        __sum.k = __sum_k / count
        __sum.chi = __sum_chi
        xftf(__sum.k, __sum.chi, kweight=__kweight, window=__window, kmin=__kmin, kmax=__kmax, group=__sum)
        self.canvas_sim_chi.axes.plot(__sum.k, __sum.chi * __sum.k ** __kweight, label=f"sum", linewidth=2)
        self.canvas_sim_ft.axes.plot(__sum.r, __sum.chir_mag, label=f"sum", linewidth=2)

        self.canvas_sim_chi.axes.legend()
        self.canvas_sim_chi.draw()
        self.canvas_sim_ft.axes.legend()
        self.canvas_sim_ft.draw()