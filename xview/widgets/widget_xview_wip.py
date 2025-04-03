import os
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pkg_resources
import logging
from PyQt5 import  QtWidgets, QtCore, uic
from PyQt5.QtWidgets import QMenu,QApplication, QTreeWidget, QTreeWidgetItem, QAbstractItemView
from PyQt5.Qt import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar

from sys import platform
from pathlib import Path
import threading
from matplotlib.figure import Figure
from xas.xasproject import XASDataSet
from isstools.elements.figure_update import update_figure
from isstools.dialogs.BasicDialogs import message_box
from xas.file_io import load_binned_df_from_file, load_binned_df_and_extended_data_from_file
import copy
from xview.dialogs.FileMetadataDialog import FileMetadataDialog

from tiled.client import from_uri
from databroker.queries import TimeRange, Key
from pathlib import Path
import time as ttime
import glob
import os
import re
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from collections import defaultdict



if platform == 'darwin':
    ui_path = pkg_resources.resource_filename('xview', 'ui/ui_xview_data-mac.ui')
else:
    ui_path = pkg_resources.resource_filename('xview', 'ui/ui_xview_wip.ui')


# class FileItem(QStandardItem):
#     def __init__(self, name, path):
#         super().__init__(name)  # Set item text
#         self.setData(path, Qt.UserRole)  # Store path in UserRole
#
#     def getPath(self):
#         return self.data(Qt.UserRole)  # Retrieve path
#
#     def setPath(self, path):
#         self.setData(path, Qt.UserRole)  # Update path

class FileItem(QStandardItem):
    def __init__(self, base_name, epoch, full_path):
        super().__init__(base_name)
        self._epoch = epoch
        self._full_path = full_path

    def full_path(self):
        """Return the full file path."""
        return self._full_path

    def epoch(self):
        """Return the file's creation timestamp (epoch)."""
        return self._epoch

    def __repr__(self):
        return f"FileItem({self.text()}, epoch={self._epoch}, path={self._full_path})"

class UIXviewWIP(*uic.loadUiType(ui_path)):
    def __init__(self, db=None, parent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.db=db
        self.addCanvas()
        self.last_keys = []
        self.group_keys = []
        self.push_extract.clicked.connect(self.initialize_extraction)
        self.push_plot_data.clicked.connect(self.plot_data)
        self.tree_structure = None
        self.experimentTreeModel = QStandardItemModel()
        self.treeView_experiment.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.initialize_extraction()

    def initialize_extraction(self):

        group_key_dict = {'': None,
                          'Sample':'Sample.name',
                          'Scan': 'Mono.trajectory_name'
                          }

        self.tree_structure = {}
        self.proposal = str(self.spinBox_proposal.value())
        self.year = str(self.spinBox_year.value())
        self.cycle = str(self.spinBox_cycle.value())
        self.directory_path = f'/nsls2/data3/iss/legacy/processed/{self.year}/{self.cycle}/{self.proposal}/'

        self.current_file_list =   glob.glob(f'{self.directory_path}*.dat')

        _group_keys = [group_key_dict[self.comboBox_group1.currentText()],
                       group_key_dict[self.comboBox_group2.currentText()],
                       group_key_dict[self.comboBox_group3.currentText()]]

        self.group_keys = []
        if _group_keys[0]:
            self.group_keys.append(_group_keys[0])
            if _group_keys[1]:
                self.group_keys.append(_group_keys[1])
                if _group_keys[2]:
                    self.group_keys.append(_group_keys[2])
        print(self.group_keys)
        self.tree_structure = self.group_files_by_metadata_keys(self.directory_path,
                                                                       self.group_keys)
        self.update_display()



    def extract_metadata_keys(self, file_path, keys):
        """Extract values for a list of metadata keys from a file."""
        values = {key: None for key in keys}
        try:
            with open(file_path, 'r', errors='ignore') as f:
                for line in f:
                    if line.startswith("#") and ":" in line:
                        key, value = line[1:].split(":", 1)
                        key = key.strip()
                        if key in values:
                            values[key] = value.strip()
                        if all(values[k] is not None for k in keys):
                            break
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        return [values[k] or "UNKNOWN" for k in keys]

    def group_files_by_metadata_keys(self, directory, metadata_keys, extension=".dat"):
        """Group files by metadata keys or list them with timestamps if keys are empty."""
        total_num_files = len(glob.glob(f'{directory}*.dat'))
        files_info = []
        jj= 0
        for filename in os.listdir(directory):
            if filename.endswith(extension):
                jj = jj+1
                self.progressBar_loading.setValue(int(100*jj/total_num_files))
                full_path = os.path.join(directory, filename)
                base_name = os.path.basename(full_path)
                try:
                    epoch_time = os.path.getctime(full_path)
                except Exception:
                    epoch_time = -1
                files_info.append((base_name, epoch_time, full_path))

        if not metadata_keys:
            # Just return sorted list of files
            return sorted(files_info, key=lambda x: x[1])

        # Build nested dict if metadata keys are provided
        def nested_dict():
            return defaultdict(nested_dict)

        root = nested_dict()

        for base_name, epoch_time, full_path in files_info:
            key_values = self.extract_metadata_keys(full_path, metadata_keys)
            current = root
            for key in key_values[:-1]:
                current = current[key]
            current.setdefault(key_values[-1], []).append((base_name, epoch_time, full_path))

        # Recursively convert and sort
        def recursive_sort(d):
            if isinstance(d, dict):
                return {k: recursive_sort(v) for k, v in d.items()}
            elif isinstance(d, list):
                return sorted(d, key=lambda x: x[1])
            else:
                return d

        import json
        return recursive_sort(json.loads(json.dumps(root)))


    def populate_tree_model(self, tree_model, data, headers=['Files']):
        """
        Populate a QTreeModel using nested dicts and FileItem leaf nodes.
        """

        def add_items(parent_item, value):
            if isinstance(value, dict):
                for key, sub_value in value.items():
                    key_item = QStandardItem(str(key))
                    parent_item.appendRow(key_item)
                    add_items(key_item, sub_value)
            elif isinstance(value, list):
                for base_name, epoch, full_path in value:
                    file_item = FileItem(base_name, epoch, full_path)
                    parent_item.appendRow(file_item)

        tree_model.clear()
        tree_model.setHorizontalHeaderLabels(headers)
        root_item = tree_model.invisibleRootItem()
        add_items(root_item, data)

    def update_display(self):
        self.experimentTreeModel.clear()
        self.populate_tree_model(self.experimentTreeModel, self.tree_structure)
        self.treeView_experiment.setModel(self.experimentTreeModel)
        self.treeView_experiment.expandAll()

    # def populate_model(self, tree_structure):
    #     if type(tree_structure) == list:
    #         sorted_files = sorted(tree_structure, key=lambda x: x[1])  # Sort by timestamp
    #         for filename, timestamp, path in sorted_files:
    #             file_item = FileItem(filename,path)
    #             self.experimentTreeModel.appendRow(file_item)
    #     else:
    #         for category, file_list in tree_structure.items():
    #             category_item = QStandardItem(category)
    #             sorted_files = sorted(file_list, key=lambda x: x[1])  # Sort by timestamp
    #             for filename, timestamp, spath in sorted_files:
    #                 file_item = FileItem(filename,path)
    #                 category_item.appendRow(file_item)
    #             self.experimentTreeModel.appendRow(category_item)

    def plot_data(self):
        selected_indexes = self.treeView_experiment.selectedIndexes()  # Get selected indexes
        selected_items = [self.experimentTreeModel.itemFromIndex(index) for index in selected_indexes]  # Convert to items
        # Print selected items
        path_list = [item.getPath() for item in selected_items if isinstance(item, FileItem)]
        self._populate_keys(path_list)
        keys = self._set_keys()
        self._plot(path_list,keys)




    def _set_keys(self):
        divide = True
        if self.checkBox_use_custom_channels.isChecked():
            _denominator = self.listWidget_data_denominator.selectedItems()
            _numerator = self.listWidget_data_numerator.selectedItems()
            if _denominator and _numerator:
                denominator = _denominator[0].text()
                numerator = _numerator[0].text()
                log = self.checkBox_log.isChecked()
                invert = self.checkBox_invert.isChecked()
                divide = self.checkBox_divide.isChecked()
            else:
                message_box('Warning', 'Please select numerator and denominator')
                return None
        else:
            if self.radioButton_tr.isChecked():
                numerator = 'i0'
                denominator = 'it'
                log = True
                invert = False
            elif self.radioButton_ref.isChecked():
                numerator = 'it'
                denominator = 'ir'
                log = True
                invert = False
            elif self.radioButton_fluo_pips.isChecked():
                numerator = 'it'
                denominator = 'ir'
                log = False
                invert = False
        return numerator, denominator, log,invert,divide

    # def extract_metadata_keys(file_path, keys):
    #     """Extract values for a list of metadata keys from a file."""
    #     values = {key: None for key in keys}
    #     try:
    #         with open(file_path, 'r', errors='ignore') as f:
    #             for line in f:
    #                 if line.startswith("#") and ":" in line:
    #                     key, value = line[1:].split(":", 1)
    #                     key = key.strip()
    #                     if key in values:
    #                         values[key] = value.strip()
    #                     if all(values[k] is not None for k in keys):
    #                         break
    #     except Exception as e:
    #         print(f"Error reading {file_path}: {e}")
    #     return [values[k] or "UNKNOWN" for k in keys]
    #
    # def group_files_by_metadata_keys(directory, metadata_keys, extension=".dat"):
    #     """Group files in a nested dictionary based on multiple metadata keys."""
    #     from collections import defaultdict
    #
    #     def nested_dict():
    #         return defaultdict(nested_dict)
    #
    #     root = nested_dict()
    #
    #     for filename in os.listdir(directory):
    #         if filename.endswith(extension):
    #             full_path = os.path.join(directory, filename)
    #             key_values = extract_metadata_keys(full_path, metadata_keys)
    #             current = root
    #             for key in key_values[:-1]:
    #                 current = current[key]
    #             current.setdefault(key_values[-1], []).append(filename)
    #
    #     # Convert defaultdict to regular dict
    #     import json
    #     return json.loads(json.dumps(root))

    def _populate_keys(self, path_list):
        df, header = load_binned_df_from_file(path_list[0])
        keys = df.keys()
        refined_keys = []
        for key in keys:
            if not (('timestamp' in key) or ('energy' in key)):
                refined_keys.append(key)

        self.keys = refined_keys
        if self.keys != self.last_keys:
            self.last_keys = self.keys
            self.listWidget_data_numerator.clear()
            self.listWidget_data_denominator.clear()
            self.listWidget_data_numerator.addItems(self.keys)
            self.listWidget_data_denominator.addItems(self.keys)

    def _plot(self, path_list,keys):
        spectra = []
        if keys:
            update_figure([self.figure_data.ax], self.toolbar, self.canvas)
            _numerator_key, _denominator_key, _log_key, _invert_key, _divide_key = keys
            for path in path_list:
                df, header = load_binned_df_from_file(path)
                energy_key = self.get_energy_key(df)
                numerator_column = np.array(df[_numerator_key])
                denominator_column = np.array(df[_denominator_key])
                if _divide_key is True:
                    mu =numerator_column/denominator_column
                    mu_label = f'{_numerator_key}/{_denominator_key}'
                else:
                    mu =numerator_column
                    mu_label = f'{_numerator_key}'
                if _log_key:
                    mu = np.log(mu)
                    mu_label= f'ln ({mu_label})'

                if _invert_key:
                    mu = -mu
                    mu_label = f'-{y_label}'

                try:
                    energy = df[energy_key]
                except:
                    energy = np.arange(mu.size)


                self.figure_data.ax.plot(energy, mu)

            self.parent.set_figure(self.figure_data.ax, self.canvas, label_x='Energy (eV)', label_y='None')

            self.figure_data.ax.set_xlabel('Energy (eV)')
            self.figure_data.ax.set_ylabel(mu_label)

        self.figure_data.ax.legend()
        self.figure_data.tight_layout()
        self.canvas.draw_idle()





    def addCanvas(self):
        self.figure_data = Figure()
        #self.figure_data.set_facecolor(color='#E2E2E2')
        self.figure_data.ax = self.figure_data.add_subplot(111)
        self.canvas = FigureCanvas(self.figure_data)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.resize(1, 10)
        self.layout_plot_data.addWidget(self.toolbar)
        self.layout_plot_data.addWidget(self.canvas)
        self.figure_data.tight_layout()
        self.canvas.draw()


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++






    def xas_data_context_menu(self,QPos):
        menu = QMenu()
        plot_action = menu.addAction("&Plot")
        add_to_project_action = menu.addAction("&Add to project")
        show_metadata_action = menu.addAction("&Show file metadata")
        # merge_action = menu.addAction("&Add to project")
        parentPosition = self.list_data.mapToGlobal(QtCore.QPoint(0, 0))
        menu.move(parentPosition+QPos)
        action = menu.exec_()
        if action == plot_action:
            self.plot_xas_data()
        elif action == add_to_project_action:
            self.add_data_to_project()
        elif action == show_metadata_action:
            self.show_file_metadata()

    def show_file_metadata(self):
        selected_items = (self.list_data.selectedItems())
        for i in selected_items:
            path = f'{self.working_folder}/{i.text()}'
            _, header = load_binned_df_from_file(path)
            self.file_md_widget = FileMetadataDialog(path, header, parent=self)
            self.file_md_widget.show()



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

            self.file_list = [f for f in os.listdir(self.working_folder) if f.endswith('.dat') or f.endswith('mu')]

            if self.comboBox_sort_files_by.currentText() == 'Name':
                self.file_list.sort()
            elif self.comboBox_sort_files_by.currentText() == 'Time':
                self.file_list.sort(key=lambda x: os.path.getmtime('{}/{}'.format(self.working_folder, x)))

                self.file_list.reverse()
            self.list_data.addItems(self.file_list)

    def select_files_to_plot(self):
        df, header = load_binned_df_from_file(f'{self.working_folder}/{self.list_data.currentItem().text()}')
        keys = df.keys()
        refined_keys = []
        for key in keys:
            if not (('timestamp' in key) or ('energy' in key)):
                refined_keys.append(key)

        self.keys = refined_keys
        if self.keys != self.last_keys:
            self.last_keys = self.keys

            self.listWidget_data_numerator.clear()
            self.listWidget_data_denominator.clear()
            self.listWidget_data_numerator.addItems(self.keys)
            self.listWidget_data_denominator.addItems(self.keys)



    def get_energy_key(self, df):
        energy_key = ''
        for key in ['johann_main_crystal_motor_cr_main_roll',
                    'johann_aux2_crystal_motor_cr_aux2_roll',
                    'johann_aux3_crystal_motor_cr_aux3_roll',
                    'johann_aux4_crystal_motor_cr_aux4_roll',
                    'johann_aux5_crystal_motor_cr_aux5_roll',
                    'energy', 'timestamp',]:
            if key in df.keys():
                energy_key = key
                break
        if energy_key != 'energy':
            print(f'x axis column data is taken from {energy_key}')
        return energy_key


    def plot_xas_data(self):
        selected_items = (self.list_data.selectedItems())
        update_figure([self.figure_data.ax], self.toolbar, self.canvas)
        if not(self.listWidget_data_denominator.selectedItems() and self.listWidget_data_numerator.selectedItems()):
            message_box('Warning','Please select numerator and denominator')
            return


            # energy_key = key


        handles = []

        for i in selected_items:
            path = f'{self.working_folder}/{i.text()}'
            print(path)
            df, header = load_binned_df_from_file(path)

            energy_key = self.get_energy_key(df)

            denominator_name = self.listWidget_data_denominator.selectedItems()[0].text()
            numerators_names = [b.text() for b in self.listWidget_data_numerator.selectedItems()]

            numerators =[]
            for numerator_name in numerators_names:
                numerators.append(np.array(df[numerator_name]))

            denominator = np.array(df[denominator_name])
            spectra = []
            y_label = ''
            for numerator, numerator_name in zip(numerators, numerators_names):
                if self.checkBox_ratio.checkState():
                    mu_channel = f'{numerator_name}/{denominator_name}'

                    spectra.append(numerator/denominator)
                else:
                    mu_channel = f'{numerator_name}'
                    spectra.append(numerator)
                y_label += mu_channel
            for spectrum in spectra:
                if self.checkBox_log_bin.checkState():
                    spectrum = np.log(spectrum)
                    y_label = f'ln ({y_label})'
                if self.checkBox_inv_bin.checkState():
                    spectrum = -spectrum
                    y_label = f'- {y_label}'
                fname = i.text()
                try:
                    energy = df[energy_key]
                except:
                    energy = np.arange(spectrum.size)
                self.figure_data.ax.plot(energy, spectrum, label='.'.join(fname.split('.')[:-1]) + ' ' + mu_channel)

            self.parent.set_figure(self.figure_data.ax,self.canvas,label_x='Energy (eV)', label_y=y_label)

            self.figure_data.ax.set_xlabel('Energy (eV)')
            self.figure_data.ax.set_ylabel(y_label)
            # last_trace = self.figure_data.ax.get_lines()[len(self.figure_data.ax.get_lines()) - 1]
            # patch = mpatches.Patch(color=last_trace.get_color(), label=i.text())
            # handles.append(patch)

        self.figure_data.ax.legend()
        self.figure_data.tight_layout()
        self.canvas.draw_idle()

    # def merge_files_and_save(self):
    #     selected_items = self.list_data.selectedItems()
    #     if selected_items != []:
    #         mu_t = []
    #         mu_f = []
    #         mu_r = []
    #         file_str = ''
    #
    #         energy = None
    #
    #         for item in selected_items:
    #             filepath = str(Path(self.working_folder) / Path(item.text()))
    #             name = Path(filepath).resolve().stem
    #             df, header = load_binned_df_from_file(filepath)
    #
    #
    #
    #
    #
    #             ds_list.append(self.parent.project._datasets[])
    #
    #         ds_list.sort(key=lambda x: x.name)
    #         mu = ds_list[0].mu
    #         mu_array = np.zeros([len(selection) + 1, len(mu)])
    #         energy_master = ds_list[0].energy
    #
    #         mu_array[0, :] = energy_master
    #         ret = self.message_box_save_datasets_as()
    #         for indx, obj in enumerate(selection):
    #             ds = ds_list[indx]
    #             energy = ds.energy
    #             if ret == 0:
    #                 yy = np.array(ds.mu)
    #                 keys = '# energy(eV), mu(E)\n'
    #             elif ret == 1:
    #                 yy = ds.norm
    #                 keys = '# energy(eV), normalized mu(E)\n'
    #             elif ret == 2:
    #                 yy = ds.flat
    #                 keys = '# energy(eV), flattened normalized mu(E)\n'
    #
    #             yy = np.interp(energy_master, energy, yy)
    #             mu_array[indx + 1, :] = yy
    #             md.append(ds.name)
    #
    #         self.mu_array = mu_array
    #         options = QtWidgets.QFileDialog.DontUseNativeDialog
    #         filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save XAS project',
    #                                                             self.parent.widget_data.working_folder,
    #                                                             'XAS dataset (*.dat)', options=options)
    #         if filename:
    #             if Path(filename).suffix != '.xas':
    #                 filename = filename + '.xas'
    #             print(filename)
    #             filelist = "{}".format("\n".join(md[0:]))
    #             separator = '\n #______________________________________________________\n'
    #
    #             header = '{} {} {}'.format(filelist, separator, keys)
    #             fid = open(filename, 'w')
    #             np.savetxt(fid, np.transpose(mu_array), header=header)
    #             fid.close()


    # def merge_xas_data(self):
    #     selected_items = (self.list_data.selectedItems())
    #     energy_key = 'energy'
    #     i0_key, it_key, ir_key, if_key = 'i0', 'it', 'ir', 'iff'
    #
    #
    #
    #     mus_array = []
    #     for i, item in enumerate(selected_items):
    #
    #         path = f'{self.working_folder}/{item.text()}'
    #         print('merging', path)
    #         df, header = load_binned_df_from_file(path)
    #         if i == 0:
    #             enregy_master = df[energy_key]
    #             mus_array_all =




    def add_data_to_project(self):
        if not(self.listWidget_data_denominator.selectedItems() and self.listWidget_data_numerator.selectedItems()):
            message_box('Warning', 'Please select numerator and denominator')
            return

        files = [item.text() for item in self.list_data.selectedItems()]
        # files.sort()
        ds_first = None
        for file in files:
            filepath = str(Path(self.working_folder) / Path(file))
            name = Path(filepath).resolve().stem

            if self.checkBox_load_extended_data.isChecked():
                df, ext_data, header = load_binned_df_and_extended_data_from_file(filepath)
            else:
                df, header = load_binned_df_from_file(filepath)
                ext_data = None

            md = {}
            try:
                uid_idx1 = header.find('Scan.uid:') + 10
                uid_idx2 = header.find('\n', header.find('Scan.uid:'))
                uid = header[uid_idx1: uid_idx2]
                md = self.db[uid]['start']
            except KeyError:
                try:
                    uid = header[header.find('UID:') + 5:header.find('\n', header.find('UID:'))]
                    md = self.db[uid]['start']
                except:
                    pass

            if md == {}:
                print('Metadata not found')

            # df = df.sort_values('energy')
            denominator_name = self.listWidget_data_denominator.selectedItems()[0].text()
            numerators_names = [b.text() for b in self.listWidget_data_numerator.selectedItems()]

            numerators = []
            for numerator_name in numerators_names:
                numerators.append(np.array(df[numerator_name]))
            denominator = np.array(df[denominator_name])

            if denominator_name == 'i0':
                denominator_sign = -1
            else:
                denominator_sign = 1

            if ext_data is not None:
                for k in ext_data.keys():
                    if k != 'data_kind':
                        if type(ext_data[k]) == dict:
                            for sub_k in ext_data[k].keys():
                                axes = tuple(i for i in range(1, len(ext_data[k][sub_k].shape)))
                                if len(axes) > 0:
                                    ext_data[k][sub_k] /= (np.expand_dims(denominator, axes) * denominator_sign)
                        else:
                            axes = tuple(i for i in range(1, len(ext_data[k].shape)))
                            ext_data[k] /= (np.expand_dims(denominator, axes) * denominator_sign)

            energy_key = self.get_energy_key(df)
            energy = df[energy_key]

            for numerator, numerator_name in zip(numerators, numerators_names):
                if self.checkBox_ratio.checkState():
                    spectrum = (numerator / denominator)
                    mu_channel = f'{numerator_name}-{denominator_name}'
                else:
                    spectrum = numerator
                    mu_channel = f'{numerator_name}-{denominator_name}'

                if self.checkBox_log_bin.checkState():
                    spectrum = np.log(spectrum)
                if self.checkBox_inv_bin.checkState():
                    spectrum = -spectrum
                try:
                    df_norm = {}
                    df_norm['energy'] = energy
                    df_norm['mut'] = -np.log(df['it'].values / df['i0'].values)
                    df_norm['muf'] = df['iff'].values / df['i0'].values
                    df_norm['mur'] = -np.log(df['ir'].values / df['it'].values)
                    df_norm = pd.DataFrame(df_norm)
                except:
                    df_norm = None
                # attempt to add dictionary
                #md['mu_channel']= mu_channel
                #print(f'Channel {mu_channel}')
                if ds_first is None:
                    ds = XASDataSet(name=(f'{name} {mu_channel}'), md=md, energy=energy, mu=spectrum, filename=filepath,
                                datatype='experiment', ext_data=ext_data, df=df_norm)
                    ds_first = ds
                # print('make first dataset')
                else:
                    ds = XASDataSet(name=(f'{name} {mu_channel}'), md=md, energy=energy, mu=spectrum, filename=filepath,
                                datatype='experiment', process=False, xasdataset=ds_first, ext_data=ext_data, df=df_norm)
                # print('copying parameters from the first dataset')

            # print('dataset energy id', ds.energy)
                ds.header = header
                self.parent.project.append(ds)
                self.parent.statusBar().showMessage('Scans added to the project successfully')



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





