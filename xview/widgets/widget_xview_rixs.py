import os
import matplotlib.patches as mpatches
import numpy as np
import pkg_resources
import copy
from PyQt5 import  QtWidgets, QtCore, uic
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QMenu
from PyQt5.Qt import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar

from sys import platform
from pathlib import Path

from matplotlib.figure import Figure
from xas.xasproject import XASDataSet
from isstools.elements.figure_update import update_figure
from isstools.dialogs.BasicDialogs import message_box
from xas.file_io import load_binned_df_from_file

from xas.spectrometer import parse_rixs_scan, parse_rixslog_scan
import h5py


if platform == 'darwin':
    ui_path = pkg_resources.resource_filename('xview', 'ui/ui_xview_data-mac.ui')
else:
    ui_path = pkg_resources.resource_filename('xview', 'ui/ui_xview_rixs.ui')


class UIXviewRIXS(*uic.loadUiType(ui_path)):
    def __init__(self, db=None, parent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)


        self.db = db
        self.parent = parent
        self.push_select_folder.clicked.connect(self.select_working_folder)
        self.push_refresh_folder.clicked.connect(self.get_file_list)

        self.push_parse_data.clicked.connect(self.parse_rixs_scan)
        self.push_plot_data.clicked.connect(self.plot_rixs_data)

        self.comboBox_sort_files_by.addItems(['Time','Name'])
        self.comboBox_sort_files_by.currentIndexChanged.connect((self.get_file_list))

        self.comboBox_data_numerator.currentIndexChanged.connect(self.update_current_numerator)
        self.comboBox_data_denominator.currentIndexChanged.connect(self.update_current_denominator)
        self.comboBox_data_bkg.currentIndexChanged.connect(self.update_current_bkg)

        self.list_data.itemSelectionChanged.connect(self.select_files_to_plot)
        # self.push_add_to_project.clicked.connect(self.add_data_to_project)
        self.list_data.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_data.customContextMenuRequested.connect(self.xas_data_context_menu)

        self.list_data.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.addCanvas()
        self.keys = []
        self.last_keys = []
        self.current_plot_in = ''
        self.binned_data = []
        self.last_numerator= ''
        self.last_denominator = ''
        self.last_bkg = ''
        # Persistent settings
        self.settings = QSettings('ISS Beamline', 'Xview')
        self.working_folder = self.settings.value('working_folder', defaultValue='/GPFS/xf08id/User Data', type=str)

        if self.working_folder != '/GPFS/xf08id/User Data':
            self.label_working_folder.setText(self.working_folder)
            self.label_working_folder.setToolTip(self.working_folder)
            self.get_file_list()

    def xas_data_context_menu(self,QPos):
        menu = QMenu()
        plot_action = menu.addAction("&Plot")
        # add_to_project_action = menu.addAction("&Add to project")
        # merge_action = menu.addAction("&Add to project")
        parentPosition = self.list_data.mapToGlobal(QtCore.QPoint(0, 0))
        menu.move(parentPosition+QPos)
        action = menu.exec_()
        if action == plot_action:
            self.plot_rixs_data()
        # elif action == add_to_project_action:
        #     self.add_data_to_project()

    def addCanvas(self):
        self.figure_rixs = Figure()
        #self.figure_data.set_facecolor(color='#E2E2E2')
        self.figure_rixs.ax = self.figure_rixs.add_subplot(111)
        self.canvas = FigureCanvas(self.figure_rixs)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.resize(1, 10)
        self.layout_plot_rixs.addWidget(self.toolbar)
        self.layout_plot_rixs.addWidget(self.canvas)
        self.figure_rixs.tight_layout()
        self.canvas.draw()

    def select_working_folder(self):
        self.working_folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a folder", self.working_folder,
                                                                        QtWidgets.QFileDialog.ShowDirsOnly)
        if self.working_folder:
            self.set_working_folder()

    def set_working_folder(self):
        self.settings.setValue('working_folder', self.working_folder)
        if len(self.working_folder) > 50:
            self.label_working_folder.setText(self.working_folder[1:20] + '...' + self.working_folder[-30:])
        else:
            self.label_working_folder.setText(self.working_folder)
        self.get_file_list()

    def get_file_list(self):
        if self.working_folder:
            self.list_data.clear()

            self.file_list = [f for f in os.listdir(self.working_folder) if f.endswith('.uids')]

            if self.comboBox_sort_files_by.currentText() == 'Name':
                self.file_list.sort()
            elif self.comboBox_sort_files_by.currentText() == 'Time':
                self.file_list.sort(key=lambda x: os.path.getmtime('{}/{}'.format(self.working_folder, x)))

                self.file_list.reverse()
            self.list_data.addItems(self.file_list)

    def select_files_to_plot(self):
        current_file = f'{self.working_folder}/{self.list_data.currentItem().text()}'
        f = h5py.File(current_file, 'r')
        uid_herfds = list(f.keys())
        f.close()
        hdr = self.db[uid_herfds[0]]
        path = hdr.start['interp_filename']
        df, header = load_binned_df_from_file(path)

        keys = df.keys()
        refined_keys = []
        for key in keys:
            if not (('timestamp' in key) or ('energy' in key)):
                refined_keys.append(key)
        self.keys = refined_keys
        if self.keys != self.last_keys:
            self.last_keys = self.keys
            self.comboBox_data_numerator.clear()
            self.comboBox_data_bkg.clear()
            self.comboBox_data_denominator.clear()
            self.comboBox_data_numerator.insertItems(0, self.keys)
            self.comboBox_data_bkg.insertItems(0, self.keys)
            self.comboBox_data_denominator.insertItems(0, self.keys)
            if self.last_numerator!= '' and self.last_numerator in self.keys:
                indx = self.comboBox_data_numerator.findText(self.last_numerator)
                self.comboBox_data_numerator.setCurrentIndex(indx)
            if self.last_denominator!= '' and self.last_denominator in self.keys:
                indx = self.comboBox_data_denominator.findText(self.last_denominator)
                self.comboBox_data_denominator.setCurrentIndex(indx)
            if self.last_bkg!= '' and self.last_bkg in self.keys:
                indx = self.comboBox_data_bkg.findText(self.last_bkg)
                self.comboBox_data_bkg.setCurrentIndex(indx)

    def update_current_numerator(self):
        self.last_numerator= self.comboBox_data_numerator.currentText()
        # print(f'Chanhin last num to {self.last_numerator}')

    def update_current_bkg(self):
        self.last_bkg= self.comboBox_data_numerator.currentText()

    def update_current_denominator(self):
        self.last_denominator= self.comboBox_data_denominator.currentText()
        # print(f'I am there {self.last_denominator}')

    def parse_rixs_scan(self):
        selected_items = (self.list_data.selectedItems())
        update_figure([self.figure_rixs.ax], self.toolbar, self.canvas)
        path = f'{self.working_folder}/{selected_items[0].text()}'
        self.rixs_dict = parse_rixslog_scan(self.db, path)

        self.process_rixs_dict()
        self.doubleSpinBox_contourf_vmin.setValue(self._plot_data.min())
        self.doubleSpinBox_contourf_vmax.setValue(np.median(self._plot_data))


    def process_rixs_dict(self):
        self._energy_in = self.rixs_dict['energy_in']
        self._energy_out = self.rixs_dict['energy_out']
        self._plot_data = copy.deepcopy(self.rixs_dict[self.last_numerator])
        if self.checkBox_bkg_subtr.checkState():
            self._plot_data -= self.rixs_dict[self.last_bkg]
        if self.checkBox_ratio.checkState():
            self._plot_data /= self.rixs_dict[self.last_denominator]
        if self.checkBox_inv_bin.checkState():
            self._plot_data *= -1




    def plot_rixs_data(self):
        self.process_rixs_dict()
        n = self.spinBox_contourf_n.value()
        vmin = self.doubleSpinBox_contourf_vmin.value()
        vmax = self.doubleSpinBox_contourf_vmax.value()
        _plot_data_ = self._plot_data.copy()
        _plot_data_[_plot_data_ < vmin] = vmin
        _plot_data_[_plot_data_ > vmax] = vmax

        self.figure_rixs.ax.contourf(self._energy_in, self._energy_out, self._plot_data.T, n, vmin=vmin, vmax=vmax)
        self.figure_rixs.ax.set_xlabel('Incident energy, eV')
        self.figure_rixs.ax.set_ylabel('Emission energy, eV')
        self.figure_rixs.tight_layout()
        self.canvas.draw_idle()

        # if self.comboBox_data_numerator.currentText() == -1 or self.comboBox_data_denominator.currentText() == -1:
        #     message_box('Warning','Please select numerator and denominator')
        #     return

        # self.last_numerator = self.comboBox_data_numerator.currentText()
        # self.last_denominator = self.comboBox_data_denominator.currentText()
        #
        # energy_key = 'energy'
        #
        # handles = []
        #
        # for i in selected_items:
        #     path = f'{self.working_folder}/{i.text()}'
        #     print(path)
        #     df, header = load_binned_df_from_file(path)
        #     numer = np.array(df[self.comboBox_data_numerator.currentText()])
        #     denom = np.array(df[self.comboBox_data_denominator.currentText()])
        #     if self.checkBox_ratio.checkState():
        #         y_label = (f'{self.comboBox_data_numerator.currentText()} / '
        #                    f'{self.comboBox_data_denominator.currentText()}')
        #         spectrum = numer/denom
        #     else:
        #         y_label = (f'{self.comboBox_data_numerator.currentText()}')
        #         spectrum = numer
        #     if self.checkBox_log_bin.checkState():
        #         spectrum = np.log(spectrum)
        #         y_label = f'ln ({y_label})'
        #     if self.checkBox_inv_bin.checkState():
        #         spectrum = -spectrum
        #         y_label = f'- {y_label}'
        #
        #     self.figure_data.ax.plot(df[energy_key], spectrum)
        #     self.parent.set_figure(self.figure_data.ax,self.canvas,label_x='Energy (eV)', label_y=y_label)
        #
        #     self.figure_data.ax.set_xlabel('Energy (eV)')
        #     self.figure_data.ax.set_ylabel(y_label)
        #     last_trace = self.figure_data.ax.get_lines()[len(self.figure_data.ax.get_lines()) - 1]
        #     patch = mpatches.Patch(color=last_trace.get_color(), label=i.text())
        #     handles.append(patch)

        #self.figure_data.ax.legend(handles=handles)
        self.figure_rixs.tight_layout()
        self.canvas.draw_idle()


    def add_data_to_project(self):
        if self.comboBox_data_numerator.currentText() != -1 and self.comboBox_data_denominator.currentText() != -1:
            for item in self.list_data.selectedItems():
                filepath = str(Path(self.working_folder) / Path(item.text()))

                name = Path(filepath).resolve().stem
                df, header = load_binned_df_from_file(filepath)
                uid = header[header.find('UID:')+5:header.find('\n', header.find('UID:'))]


                try:
                    md = self.db[uid]['start']
                except:
                    print('Metadata not found')
                    md={}

                df = df.sort_values('energy')
                num_key = self.comboBox_data_numerator.currentText()
                den_key = self.comboBox_data_denominator.currentText()
                mu = df[num_key] / df[den_key]

                if self.checkBox_log_bin.checkState():
                    mu = np.log(mu)
                if self.checkBox_inv_bin.checkState():
                    mu = -mu
                mu=np.array(mu)

                ds = XASDataSet(name=name,md=md,energy=df['energy'],mu=mu, filename=filepath,datatype='experiment')
                ds.header = header
                self.parent.project.append(ds)
                self.parent.statusBar().showMessage('Scans added to the project successfully')
        else:
            message_box('Error', 'Select numerator and denominator columns')


    def set_selection(self, name):
        index = 0
        names = []
        for index in range(self.list_data.count()):
            names.append(self.list_data.item(index).text().split('.')[0])
        try:
            index = names.index(name)
            print(index)
        except:
            print('not found')
        if index:
            self.list_data.setCurrentRow(index)





