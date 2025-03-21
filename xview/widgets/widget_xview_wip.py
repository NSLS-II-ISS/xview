# import os
# import matplotlib.patches as mpatches
# import numpy as np
# import pandas as pd
# import pkg_resources
#
# from PyQt5 import  QtWidgets, QtCore, uic
# from PyQt5.QtCore import QSettings
# from PyQt5.QtWidgets import QMenu
# from PyQt5.Qt import Qt
# from PyQt5.QtGui import QStandardItemModel, QStandardItem
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
#     NavigationToolbar2QT as NavigationToolbar
#
# from sys import platform
# from pathlib import Path
#
# from matplotlib.figure import Figure
# from xas.xasproject import XASDataSet
# from isstools.elements.figure_update import update_figure
# from isstools.dialogs.BasicDialogs import message_box
# from xas.file_io import load_binned_df_from_file, load_binned_df_and_extended_data_from_file
# import copy
# from xview.dialogs.FileMetadataDialog import FileMetadataDialog
#
# from tiled.client import from_uri
# from databroker.queries import TimeRange, Key
#
# if platform == 'darwin':
#     ui_path = pkg_resources.resource_filename('xview', 'ui/ui_xview_data-mac.ui')
# else:
#     ui_path = pkg_resources.resource_filename('xview', 'ui/ui_xview_wip.ui')
#
# class ExperimentTreeModel(QStandardItemModel):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setHorizontalHeaderLabels(["Name"])
#         self.populate_model()
#
#     def populate_model(self):
#         # Create a root item
#         self.root_item = self.invisibleRootItem()
#
#     def add_item(self, name, parent_item=None):
#         """Add a new item to the model."""
#         new_item = QStandardItem(name)
#         new_item.setEditable(False)
#
#         if parent_item:
#             parent_item.appendRow(new_item)  # Add to selected parent
#         else:
#             self.root_item.appendRow(new_item)  # Add to the root
#
#         return new_item
#
# class UIXviewWIP(*uic.loadUiType(ui_path)):
#     def __init__(self, db=None, parent=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.setupUi(self)
#         self.tiled_catalog = None
#         self.current_catalog = None
#         self.uid_list = None
#         self.push_extract.clicked.connect(self.extract_docs)
#         self.proposal = None
#         self.year = None
#         self.cycle = None
#         self.cycle_def = {'1': ['01','01','04','30'],
#                           '2':['05','01','08','31'],
#                           '3':['09','01','12','31'],}
#         self.db=db
#
#         self.treeView_experiment.model = ExperimentTreeModel()
#
#
#     def extract_docs(self):
#         if self.tiled_catalog is None:
#             self.tiled_catalog = from_uri("https://tiled.nsls2.bnl.gov")["iss"]["raw"]
#         else:
#             print('?????????????????????')
#
#         self.proposal = str(self.spinBox_proposal.value())
#         self.year = str(self.spinBox_year.value())
#         self.cycle = str(self.spinBox_cycle.value())
#         start_date = f"{self.year}-{self.cycle_def[self.cycle][0]}-{self.cycle_def[self.cycle][1]}"
#         end_date   = f"{self.year}-{self.cycle_def[self.cycle][2]}-{self.cycle_def[self.cycle][3]}"
#
#         date_limited_c = self.tiled_catalog.search(TimeRange(since=start_date, until=end_date))
#
#         self.current_catalog = date_limited_c.search(Key("proposal") == self.proposal)
#         self.uid_list = list(self.current_catalog)
#
#         sample_list = []
#         for uid in self.uid_list:
#             if 'sample_name' in self.db[uid].start.keys():
#                 sample_list.append(self.db[uid].start['sample_name'])
#         self.unique_sample_list = list(set(sample_list))
#         self.populate_experiment_tree()
#
#     def populate_experiment_tree(self):
#         for sample in self.unique_sample_list:
#             self.treeView_experiment.model.add_item(sample)
#             self.treeView_experiment.expandAll()
#
#
#
#
#
#
#     def xas_data_context_menu(self,QPos):
#         menu = QMenu()
#         plot_action = menu.addAction("&Plot")
#         add_to_project_action = menu.addAction("&Add to project")
#         show_metadata_action = menu.addAction("&Show file metadata")
#         # merge_action = menu.addAction("&Add to project")
#         parentPosition = self.list_data.mapToGlobal(QtCore.QPoint(0, 0))
#         menu.move(parentPosition+QPos)
#         action = menu.exec_()
#         if action == plot_action:
#             self.plot_xas_data()
#         elif action == add_to_project_action:
#             self.add_data_to_project()
#         elif action == show_metadata_action:
#             self.show_file_metadata()
#
#     def show_file_metadata(self):
#         selected_items = (self.list_data.selectedItems())
#         for i in selected_items:
#             path = f'{self.working_folder}/{i.text()}'
#             _, header = load_binned_df_from_file(path)
#             self.file_md_widget = FileMetadataDialog(path, header, parent=self)
#             self.file_md_widget.show()
#
#     def addCanvas(self):
#         self.figure_data = Figure()
#         #self.figure_data.set_facecolor(color='#E2E2E2')
#         self.figure_data.ax = self.figure_data.add_subplot(111)
#         self.canvas = FigureCanvas(self.figure_data)
#         self.toolbar = NavigationToolbar(self.canvas, self)
#         self.toolbar.resize(1, 10)
#         self.layout_plot_data.addWidget(self.toolbar)
#         self.layout_plot_data.addWidget(self.canvas)
#         self.figure_data.tight_layout()
#         self.canvas.draw()
#
#     def select_working_folder(self):
#         self.working_folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a folder", self.working_folder,
#                                                                         QtWidgets.QFileDialog.ShowDirsOnly)
#         if self.working_folder:
#             self.set_working_folder()
#
#     def set_working_folder(self):
#         self.settings.setValue('working_folder', self.working_folder)
#         if len(self.working_folder) > 50:
#             self.label_working_folder.setText(self.working_folder[1:20] + '...' + self.working_folder[-30:])
#         else:
#             self.label_working_folder.setText(self.working_folder)
#         self.get_file_list()
#
#     def get_file_list(self):
#         if self.working_folder:
#             self.list_data.clear()
#
#             self.file_list = [f for f in os.listdir(self.working_folder) if f.endswith('.dat') or f.endswith('mu')]
#
#             if self.comboBox_sort_files_by.currentText() == 'Name':
#                 self.file_list.sort()
#             elif self.comboBox_sort_files_by.currentText() == 'Time':
#                 self.file_list.sort(key=lambda x: os.path.getmtime('{}/{}'.format(self.working_folder, x)))
#
#                 self.file_list.reverse()
#             self.list_data.addItems(self.file_list)
#
#     def select_files_to_plot(self):
#         df, header = load_binned_df_from_file(f'{self.working_folder}/{self.list_data.currentItem().text()}')
#         keys = df.keys()
#         refined_keys = []
#         for key in keys:
#             if not (('timestamp' in key) or ('energy' in key)):
#                 refined_keys.append(key)
#
#         self.keys = refined_keys
#         if self.keys != self.last_keys:
#             self.last_keys = self.keys
#
#             self.listWidget_data_numerator.clear()
#             self.listWidget_data_denominator.clear()
#             self.listWidget_data_numerator.addItems(self.keys)
#             self.listWidget_data_denominator.addItems(self.keys)
#
#     def get_energy_key(self, df):
#         energy_key = ''
#         for key in ['johann_main_crystal_motor_cr_main_roll',
#                     'johann_aux2_crystal_motor_cr_aux2_roll',
#                     'johann_aux3_crystal_motor_cr_aux3_roll',
#                     'johann_aux4_crystal_motor_cr_aux4_roll',
#                     'johann_aux5_crystal_motor_cr_aux5_roll',
#                     'energy', 'timestamp',]:
#             if key in df.keys():
#                 energy_key = key
#                 break
#         if energy_key != 'energy':
#             print(f'x axis column data is taken from {energy_key}')
#         return energy_key
#
#
#     def plot_xas_data(self):
#         selected_items = (self.list_data.selectedItems())
#         update_figure([self.figure_data.ax], self.toolbar, self.canvas)
#         if not(self.listWidget_data_denominator.selectedItems() and self.listWidget_data_numerator.selectedItems()):
#             message_box('Warning','Please select numerator and denominator')
#             return
#
#
#             # energy_key = key
#
#
#         handles = []
#
#         for i in selected_items:
#             path = f'{self.working_folder}/{i.text()}'
#             print(path)
#             df, header = load_binned_df_from_file(path)
#
#             energy_key = self.get_energy_key(df)
#
#             denominator_name = self.listWidget_data_denominator.selectedItems()[0].text()
#             numerators_names = [b.text() for b in self.listWidget_data_numerator.selectedItems()]
#
#             numerators =[]
#             for numerator_name in numerators_names:
#                 numerators.append(np.array(df[numerator_name]))
#
#             denominator = np.array(df[denominator_name])
#             spectra = []
#             y_label = ''
#             for numerator, numerator_name in zip(numerators, numerators_names):
#                 if self.checkBox_ratio.checkState():
#                     mu_channel = f'{numerator_name}/{denominator_name}'
#
#                     spectra.append(numerator/denominator)
#                 else:
#                     mu_channel = f'{numerator_name}'
#                     spectra.append(numerator)
#                 y_label += mu_channel
#             for spectrum in spectra:
#                 if self.checkBox_log_bin.checkState():
#                     spectrum = np.log(spectrum)
#                     y_label = f'ln ({y_label})'
#                 if self.checkBox_inv_bin.checkState():
#                     spectrum = -spectrum
#                     y_label = f'- {y_label}'
#                 fname = i.text()
#                 try:
#                     energy = df[energy_key]
#                 except:
#                     energy = np.arange(spectrum.size)
#                 self.figure_data.ax.plot(energy, spectrum, label='.'.join(fname.split('.')[:-1]) + ' ' + mu_channel)
#
#             self.parent.set_figure(self.figure_data.ax,self.canvas,label_x='Energy (eV)', label_y=y_label)
#
#             self.figure_data.ax.set_xlabel('Energy (eV)')
#             self.figure_data.ax.set_ylabel(y_label)
#             # last_trace = self.figure_data.ax.get_lines()[len(self.figure_data.ax.get_lines()) - 1]
#             # patch = mpatches.Patch(color=last_trace.get_color(), label=i.text())
#             # handles.append(patch)
#
#         self.figure_data.ax.legend()
#         self.figure_data.tight_layout()
#         self.canvas.draw_idle()
#
#     # def merge_files_and_save(self):
#     #     selected_items = self.list_data.selectedItems()
#     #     if selected_items != []:
#     #         mu_t = []
#     #         mu_f = []
#     #         mu_r = []
#     #         file_str = ''
#     #
#     #         energy = None
#     #
#     #         for item in selected_items:
#     #             filepath = str(Path(self.working_folder) / Path(item.text()))
#     #             name = Path(filepath).resolve().stem
#     #             df, header = load_binned_df_from_file(filepath)
#     #
#     #
#     #
#     #
#     #
#     #             ds_list.append(self.parent.project._datasets[])
#     #
#     #         ds_list.sort(key=lambda x: x.name)
#     #         mu = ds_list[0].mu
#     #         mu_array = np.zeros([len(selection) + 1, len(mu)])
#     #         energy_master = ds_list[0].energy
#     #
#     #         mu_array[0, :] = energy_master
#     #         ret = self.message_box_save_datasets_as()
#     #         for indx, obj in enumerate(selection):
#     #             ds = ds_list[indx]
#     #             energy = ds.energy
#     #             if ret == 0:
#     #                 yy = np.array(ds.mu)
#     #                 keys = '# energy(eV), mu(E)\n'
#     #             elif ret == 1:
#     #                 yy = ds.norm
#     #                 keys = '# energy(eV), normalized mu(E)\n'
#     #             elif ret == 2:
#     #                 yy = ds.flat
#     #                 keys = '# energy(eV), flattened normalized mu(E)\n'
#     #
#     #             yy = np.interp(energy_master, energy, yy)
#     #             mu_array[indx + 1, :] = yy
#     #             md.append(ds.name)
#     #
#     #         self.mu_array = mu_array
#     #         options = QtWidgets.QFileDialog.DontUseNativeDialog
#     #         filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save XAS project',
#     #                                                             self.parent.widget_data.working_folder,
#     #                                                             'XAS dataset (*.dat)', options=options)
#     #         if filename:
#     #             if Path(filename).suffix != '.xas':
#     #                 filename = filename + '.xas'
#     #             print(filename)
#     #             filelist = "{}".format("\n".join(md[0:]))
#     #             separator = '\n #______________________________________________________\n'
#     #
#     #             header = '{} {} {}'.format(filelist, separator, keys)
#     #             fid = open(filename, 'w')
#     #             np.savetxt(fid, np.transpose(mu_array), header=header)
#     #             fid.close()
#
#
#     # def merge_xas_data(self):
#     #     selected_items = (self.list_data.selectedItems())
#     #     energy_key = 'energy'
#     #     i0_key, it_key, ir_key, if_key = 'i0', 'it', 'ir', 'iff'
#     #
#     #
#     #
#     #     mus_array = []
#     #     for i, item in enumerate(selected_items):
#     #
#     #         path = f'{self.working_folder}/{item.text()}'
#     #         print('merging', path)
#     #         df, header = load_binned_df_from_file(path)
#     #         if i == 0:
#     #             enregy_master = df[energy_key]
#     #             mus_array_all =
#
#
#
#
#     def add_data_to_project(self):
#         if not(self.listWidget_data_denominator.selectedItems() and self.listWidget_data_numerator.selectedItems()):
#             message_box('Warning', 'Please select numerator and denominator')
#             return
#
#         files = [item.text() for item in self.list_data.selectedItems()]
#         # files.sort()
#         ds_first = None
#         for file in files:
#             filepath = str(Path(self.working_folder) / Path(file))
#             name = Path(filepath).resolve().stem
#
#             if self.checkBox_load_extended_data.isChecked():
#                 df, ext_data, header = load_binned_df_and_extended_data_from_file(filepath)
#             else:
#                 df, header = load_binned_df_from_file(filepath)
#                 ext_data = None
#
#             md = {}
#             try:
#                 uid_idx1 = header.find('Scan.uid:') + 10
#                 uid_idx2 = header.find('\n', header.find('Scan.uid:'))
#                 uid = header[uid_idx1: uid_idx2]
#                 md = self.db[uid]['start']
#             except KeyError:
#                 try:
#                     uid = header[header.find('UID:') + 5:header.find('\n', header.find('UID:'))]
#                     md = self.db[uid]['start']
#                 except:
#                     pass
#
#             if md == {}:
#                 print('Metadata not found')
#
#             # df = df.sort_values('energy')
#             denominator_name = self.listWidget_data_denominator.selectedItems()[0].text()
#             numerators_names = [b.text() for b in self.listWidget_data_numerator.selectedItems()]
#
#             numerators = []
#             for numerator_name in numerators_names:
#                 numerators.append(np.array(df[numerator_name]))
#             denominator = np.array(df[denominator_name])
#
#             if denominator_name == 'i0':
#                 denominator_sign = -1
#             else:
#                 denominator_sign = 1
#
#             if ext_data is not None:
#                 for k in ext_data.keys():
#                     if k != 'data_kind':
#                         if type(ext_data[k]) == dict:
#                             for sub_k in ext_data[k].keys():
#                                 axes = tuple(i for i in range(1, len(ext_data[k][sub_k].shape)))
#                                 if len(axes) > 0:
#                                     ext_data[k][sub_k] /= (np.expand_dims(denominator, axes) * denominator_sign)
#                         else:
#                             axes = tuple(i for i in range(1, len(ext_data[k].shape)))
#                             ext_data[k] /= (np.expand_dims(denominator, axes) * denominator_sign)
#
#             energy_key = self.get_energy_key(df)
#             energy = df[energy_key]
#
#             for numerator, numerator_name in zip(numerators, numerators_names):
#                 if self.checkBox_ratio.checkState():
#                     spectrum = (numerator / denominator)
#                     mu_channel = f'{numerator_name}-{denominator_name}'
#                 else:
#                     spectrum = numerator
#                     mu_channel = f'{numerator_name}-{denominator_name}'
#
#                 if self.checkBox_log_bin.checkState():
#                     spectrum = np.log(spectrum)
#                 if self.checkBox_inv_bin.checkState():
#                     spectrum = -spectrum
#                 try:
#                     df_norm = {}
#                     df_norm['energy'] = energy
#                     df_norm['mut'] = -np.log(df['it'].values / df['i0'].values)
#                     df_norm['muf'] = df['iff'].values / df['i0'].values
#                     df_norm['mur'] = -np.log(df['ir'].values / df['it'].values)
#                     df_norm = pd.DataFrame(df_norm)
#                 except:
#                     df_norm = None
#                 # attempt to add dictionary
#                 #md['mu_channel']= mu_channel
#                 #print(f'Channel {mu_channel}')
#                 if ds_first is None:
#                     ds = XASDataSet(name=(f'{name} {mu_channel}'), md=md, energy=energy, mu=spectrum, filename=filepath,
#                                 datatype='experiment', ext_data=ext_data, df=df_norm)
#                     ds_first = ds
#                 # print('make first dataset')
#                 else:
#                     ds = XASDataSet(name=(f'{name} {mu_channel}'), md=md, energy=energy, mu=spectrum, filename=filepath,
#                                 datatype='experiment', process=False, xasdataset=ds_first, ext_data=ext_data, df=df_norm)
#                 # print('copying parameters from the first dataset')
#
#             # print('dataset energy id', ds.energy)
#                 ds.header = header
#                 self.parent.project.append(ds)
#                 self.parent.statusBar().showMessage('Scans added to the project successfully')
#
#
#
#     def set_selection(self, name):
#         index = 0
#         names = []
#         for index in range(self.list_data.count()):
#             names.append(self.list_data.item(index).text().split('.')[0])
#         try:
#             index = names.index(name)
#             print(index)
#         except:
#             print('not found')
#         if index:
#             self.list_data.setCurrentRow(index)
#
#
#
#
#
