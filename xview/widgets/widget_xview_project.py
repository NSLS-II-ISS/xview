import copy
import os
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pkg_resources
from PyQt5 import QtGui, QtWidgets, QtCore, uic
from PyQt5.Qt import QSplashScreen, QObject
from PyQt5.QtCore import QSettings, QThread, pyqtSignal, QTimer, QDateTime
from PyQt5.QtWidgets import QMenu
from PyQt5.QtGui import QPixmap
from PyQt5.Qt import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from sys import platform
from pathlib import Path
from isstools.dialogs.BasicDialogs import message_box
from matplotlib.figure import Figure
from isstools.elements.figure_update import update_figure

from xas.xray import k2e, e2k
from xas.file_io import load_binned_df_from_file, dump_tiff_images
from xas.xasproject import XASDataSet
from xview.dialogs.MetadataDialog import MetadataDialog
# from xview.spectra_db.db_io import save_spectrum_to_db
from matplotlib import pyplot as plt
from os.path import expanduser
from scipy.stats import zscore



if platform == 'darwin':
    ui_path = pkg_resources.resource_filename('xview', 'ui/ui_xview_project-mac.ui')
else:
    ui_path = pkg_resources.resource_filename('xview', 'ui/ui_xview_project.ui')


class UIXviewProject(*uic.loadUiType(ui_path)):
        def __init__(self, db_proc=None,
                     parent=None, cloud_dispatcher = None, *args, **kwargs):

            super().__init__(*args, **kwargs)
            self.setupUi(self)
            self.parent = parent
            self.cloud_dispatcher = cloud_dispatcher
            self.db_proc = db_proc
            self.parent.project.datasets_changed.connect(self.update_project_list)
            self.addCanvas()
            self.label_E0.setText("E<sub>0</sub>")
            self.list_project.itemSelectionChanged.connect(self.show_ds_params)
            self.list_project.setContextMenuPolicy(Qt.CustomContextMenu)
            self.list_project.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            self.list_project.customContextMenuRequested.connect(self.xas_project_context_menu)
            self.list_project.doubleClicked.connect(self.xas_project_double_clicked)
            self.push_plot_project_in_E.clicked.connect(self.plot_project_in_E)
            self.push_plot_project_in_K.clicked.connect(self.plot_project_in_K)
            self.push_plot_project_in_R.clicked.connect(self.plot_project_in_R)

            self.push_plot_project_in_E_norm_by_max.clicked.connect(self.plot_project_in_E_norm_by_max)

            foos = [self.lineEdit_e0.textEdited,
                    self.lineEdit_preedge_lo.textEdited,
                    self.lineEdit_preedge_hi.textEdited,
                    self.lineEdit_postedge_lo.textEdited,
                    self.lineEdit_postedge_hi.textEdited,
                    self.lineEdit_postedge_nnorm.textEdited,
                    self.lineEdit_spline_lo.textEdited,
                    self.lineEdit_spline_hi.textEdited,
                    self.lineEdit_clamp_lo.textEdited,
                    self.lineEdit_clamp_hi.textEdited,
                    self.lineEdit_clamp_hi.textEdited,
                    self.lineEdit_rbkg.textEdited,
                    self.spinBox_k_weight.valueChanged,
                    self.lineEdit_k_ft_lo.textEdited,
                    self.lineEdit_k_ft_hi.textEdited]
            for foo in foos:
                foo.connect(self.update_ds_params)

            self.pushButton_e0_set.clicked.connect(self.set_ds_params_from_plot)
            self.pushButton_preedge_lo_set.clicked.connect(self.set_ds_params_from_plot)
            self.pushButton_preedge_hi_set.clicked.connect(self.set_ds_params_from_plot)
            self.pushButton_postedge_lo_set.clicked.connect(self.set_ds_params_from_plot)
            self.pushButton_postedge_hi_set.clicked.connect(self.set_ds_params_from_plot)
            self.pushButton_spline_lo_set.clicked.connect(self.set_ds_params_from_plot)
            self.pushButton_spline_hi_set.clicked.connect(self.set_ds_params_from_plot)
            self.pushButton_k_ft_lo_set.clicked.connect(self.set_ds_params_from_plot)
            self.pushButton_k_ft_hi_set.clicked.connect(self.set_ds_params_from_plot)

            self.pushButton_truncate_at_set.clicked.connect(self.set_ds_params_from_plot)

            # Push to selected/all  buttons defs
            self.pushButton_push_norm_param_to_selected.clicked.connect(self.push_param)
            self.pushButton_push_norm_param_to_all.clicked.connect(self.push_param)
            self.pushButton_push_bkg_param_to_selected.clicked.connect(self.push_param)
            self.pushButton_push_bkg_param_to_all.clicked.connect(self.push_param)
            self.pushButton_push_ft_param_to_selected.clicked.connect(self.push_param)
            self.pushButton_push_ft_param_to_all.clicked.connect(self.push_param)

            self.pushButton_truncate_below.clicked.connect(self.truncate)
            self.pushButton_truncate_above.clicked.connect(self.truncate)

            # Menu defs
            # self.action_exit.triggered.connect(self.close_app)
            # self.action_save_project.triggered.connect(self.save_xas_project)
            # self.action_open_project.triggered.connect(self.open_xas_project)
            # self.action_save_datasets_as_text.triggered.connect(self.save_xas_datasets_as_text)
            # self.action_combine_and_save_as_text.triggered.connect(self.combine_and_save_datasets_as_text)
            # self.action_merge.triggered.connect(self.merge_datasets)
            # self.action_rename.triggered.connect(self.rename_dataset)
            # self.action_remove.triggered.connect(self.remove_from_xas_project)

            self.lineEdit_to_ds_parameter_dict = {
                'lineEdit_preedge_lo':     'pre1',
                'lineEdit_preedge_hi':     'pre2',
                'lineEdit_postedge_lo':    'norm1',
                'lineEdit_postedge_hi':    'norm2',
                'lineEdit_postedge_nnorm': 'nnorm',
                'lineEdit_e0':             'e0',
                'lineEdit_spline_lo':      'kmin',
                'lineEdit_spline_hi':      'kmax',
                'lineEdit_clamp_lo':       'clamp_lo',
                'lineEdit_clamp_hi':       'clamp_hi',
                'lineEdit_rbkg':           'rbkg',
                'spinBox_k_weight':        'kweight',
                'lineEdit_truncate_at':    'truncate',
                'lineEdit_k_ft_lo':        'kmin_ft',
                'lineEdit_k_ft_hi':        'kmax_ft'
            }

            self.pushButton_set_to_lineEdit_dict = {
                'pushButton_e0_set': 'lineEdit_e0',
                'pushButton_preedge_lo_set':    'lineEdit_preedge_lo',
                'pushButton_preedge_hi_set':    'lineEdit_preedge_hi',
                'pushButton_postedge_lo_set':   'lineEdit_postedge_lo',
                'pushButton_postedge_hi_set':   'lineEdit_postedge_hi',
                'pushButton_spline_lo_set':     'lineEdit_spline_lo',
                'pushButton_spline_hi_set':     'lineEdit_spline_hi',
                'pushButton_k_ft_lo_set':       'lineEdit_k_ft_lo',
                'pushButton_k_ft_hi_set':       'lineEdit_k_ft_hi',
                'pushButton_truncate_at_set':   'lineEdit_truncate_at'
            }
            self.windows_list = [
                'hanning',
                'kaiser',
                'gaussian',
                'sine'
            ]

        def xas_project_context_menu(self, QPos):
            menu = QMenu()
            rename_action = menu.addAction("&Rename")
            merge_action = menu.addAction("&Merge")
            show_ext_data_action = menu.addAction("&Show extended data")
            remove_action = menu.addAction("&Remove")
            save_datasets_as_text_action = menu.addAction("&Save datasets as text")
            combine_and_save_datasets_as_text_action = menu.addAction("&Combine and save datasets as text")
            save_dataset_to_dropbox = menu.addAction("&Save to Dropbox")
            # save_dataset_to_database_action = menu.addAction("&Save to processed database")
            export_dataset_to_mcr_project = menu.addAction("&Add as dataset as MCR project")
            export_ref_to_mcr_project = menu.addAction("&Add as reference to MCR project")
            parentPosition = self.list_project.mapToGlobal(QtCore.QPoint(0, 0))
            menu.move(parentPosition + QPos)
            action = menu.exec_()
            if action == rename_action:
                self.rename_dataset()
            elif action == merge_action:
                self.merge_datasets()
            elif action == show_ext_data_action:
                self.show_ext_data()
            elif action == remove_action:
                self.remove_from_xas_project()
            elif action == combine_and_save_datasets_as_text_action:
                self.combine_and_save_datasets_as_text()
            elif action == save_datasets_as_text_action:
                self.save_datasets_as_text( )
            # elif action == save_dataset_to_database_action:
            #     self.save_datasets_to_database()
            elif action == save_dataset_to_dropbox:
                self.save_datasets_as_text(send_to_dropbox = True)

            elif action == export_dataset_to_mcr_project:
                self.export_dataset_to_mcr_project()
            elif action == export_ref_to_mcr_project:
                self.export_ref_to_mcr_project()

        def xas_project_double_clicked(self):
            selection = self.list_project.selectedIndexes()
            if selection != []:
                self.rename_dataset()


        def addCanvas(self):
            # XASProject Plot:
            self.figure_project = Figure()
            #self.figure_project.set_facecolor(color='#E2E2E2')
            self.figure_project.ax = self.figure_project.add_subplot(111)
            self.figure_project.ax.grid(alpha=0.4)
            self.canvas_project = FigureCanvas(self.figure_project)

            self.toolbar_project = NavigationToolbar(self.canvas_project, self)
            self.layout_plot_project.addWidget(self.canvas_project)
            self.layout_plot_project.addWidget(self.toolbar_project)
            self.figure_project.tight_layout()

            self.canvas_project.draw()
            # layout_plot_xasproject

        def push_param(self):
            self.norm_param_list = [
                'e0',
                'pre1',
                'pre2',
                'norm1',
                'norm2',
                'nnorm'
            ]

            self.bkg_param_list = [
                'kmin',
                'kmax',
                'clamp_lo',
                'clamp_hi',
                'rbkg',
                'kweight'
            ]
            self.ft_param_list = [
                'kmin_ft', 'kmax_ft'
            ]
            selection = self.list_project.selectedIndexes()
            if selection != []:
                sender = QObject()
                sender_object = sender.sender().objectName()
                index = selection[0].row()
                ds_master = self.parent.project[index]
                if sender_object == 'pushButton_push_norm_param_to_selected':
                    for indx, obj in enumerate(selection):
                        ds = self.parent.project[selection[indx].row()]
                        for param in self.norm_param_list:
                            setattr(ds, param, getattr(ds_master, param))
                if sender_object == 'pushButton_push_norm_param_to_all':
                    for indx, obj in enumerate(self.parent.project):
                        for param in self.norm_param_list:
                            setattr(self.parent.project[indx], param, getattr(ds_master, param))
                if sender_object == 'pushButton_push_bkg_param_to_selected':
                    for indx, obj in enumerate(selection):
                        ds = self.parent.project[selection[indx].row()]
                        for param in self.bkg_param_list:
                            setattr(ds, param, getattr(ds_master, param))
                if sender_object == 'pushButton_push_bkg_param_to_all':
                    for indx, obj in enumerate(self.parent.project):
                        for param in self.bkg_param_list:
                            setattr(self.parent.project[indx], param, getattr(ds_master, param))
                if sender_object == 'pushButton_push_ft_param_to_selected':
                    for indx, obj in enumerate(selection):
                        ds = self.parent.project[selection[indx].row()]
                        for param in self.ft_param_list:
                            setattr(ds, param, getattr(ds_master, param))
                if sender_object == 'pushButton_push_ft_param_to_all':
                    for indx, obj in enumerate(self.parent.project):
                        for param in self.ft_param_list:
                            setattr(self.parent.project[indx], param, getattr(ds_master, param))

        # here we begin to work on the second pre-processing tab
        def update_ds_params(self):
            sender = QObject()
            sender_object = sender.sender().objectName()
            print(sender_object)
            selection = self.list_project.selectedIndexes()
            if selection != []:
                index = selection[0].row()
                ds = self.parent.project[index]
                try:
                    self.parent.statusBar().showMessage(sender_object)
                    print(getattr(self, sender_object).text())
                    setattr(ds, self.lineEdit_to_ds_parameter_dict[sender_object],
                            float(getattr(self, sender_object).text()))
                except:
                    self.parent.statusBar().showMessage('Use numbers only')

        def set_ds_params_from_plot(self):
            sender = QObject()
            self.sender_object = sender.sender().objectName()
            self.parent.statusBar().showMessage('Click on graph or press Esc')
            self.cid = self.canvas_project.mpl_connect('button_press_event', self.mouse_press_event)

        def _disconnect_cid(self):
            if hasattr(self, 'cid'):
                self.canvas_project.mpl_disconnect(self.cid)
                delattr(self, 'cid')

        def keyPressEvent(self, event):
            if event.key() == QtCore.Qt.Key_Escape:
                self._disconnect_cid()

        def mouse_press_event(self, event):

            e_vs_k_discriminate_list = ['pushButton_spline_lo_set',
                                        'pushButton_spline_hi_set',
                                        'pushButton_k_ft_lo_set',
                                        'pushButton_k_ft_hi_set'
                                        ]

            lineEdit = getattr(self, self.pushButton_set_to_lineEdit_dict[self.sender_object])
            e0 = float(self.lineEdit_e0.text())
            if self.sender_object == 'pushButton_e0_set':
                new_value = event.xdata

            elif self.sender_object == 'pushButton_truncate_at_set':
                if self.current_plot_in == 'e':
                    new_value = event.xdata
                elif self.current_plot_in == 'k':
                    new_value = k2e(event.xdata, e0)

            elif self.sender_object in e_vs_k_discriminate_list:
                if self.current_plot_in == 'k':
                    new_value = event.xdata
                elif self.current_plot_in == 'e':
                    new_value = e2k(event.xdata, e0)
            else:
                new_value = event.xdata - e0

            lineEdit.setText('{:.1f}'.format(new_value))
            sender_object = lineEdit

            print(sender_object)
            selection = self.list_project.selectedIndexes()
            if selection != []:
                index = selection[0].row()
                ds = self.parent.project[index]
                try:
                    float(sender_object.text())
                    setattr(ds, self.lineEdit_to_ds_parameter_dict[sender_object.objectName()],
                            float(sender_object.text()))
                except:
                    print('what''s going wrong')

            self._disconnect_cid()

        def update_project_list(self, datasets):
            self.list_project.clear()
            for ds in datasets:
                self.list_project.addItem(ds.name)


        def show_ds_params(self):
            if self.list_project.selectedIndexes():
                index = self.list_project.selectedIndexes()[0]
                ds = self.parent.project[index.row()]
                self.lineEdit_e0.setText('{:.1f}'.format(ds.e0))
                self.lineEdit_preedge_lo.setText('{:.1f}'.format(ds.pre1))
                self.lineEdit_preedge_hi.setText('{:.1f}'.format(ds.pre2))
                self.lineEdit_postedge_lo.setText('{:.1f}'.format(ds.norm1))
                self.lineEdit_postedge_hi.setText('{:.1f}'.format(ds.norm2))
                self.lineEdit_postedge_nnorm.setText('{:.1f}'.format(ds.nnorm))
                self.lineEdit_spline_lo.setText('{:.1f}'.format(ds.kmin))
                self.lineEdit_spline_hi.setText('{:.1f}'.format(ds.kmax))
                self.lineEdit_clamp_lo.setText('{:.1f}'.format(ds.clamp_lo))
                self.lineEdit_clamp_hi.setText('{:.1f}'.format(ds.clamp_hi))
                self.lineEdit_k_ft_lo.setText('{:.1f}'.format(ds.kmin_ft))
                self.lineEdit_k_ft_hi.setText('{:.1f}'.format(ds.kmax_ft))

                # Make the first selected line bold, and reset bold font for other selections
                font = QtGui.QFont()
                font.setBold(False)

                for i in range(self.list_project.count()):
                    self.list_project.item(i).setFont(font)
                font.setBold(True)
                self.list_project.item(index.row()).setFont(font)

            if self.list_project.selectedIndexes():

                indices = self.list_project.selectedIndexes()
                for index in indices:
                    self.parent.widget_statistics.list_project.item(index.row()).setSelected(True)


        def remove_from_xas_project(self):
            for index in self.list_project.selectedIndexes()[
                         ::-1]:  # [::-1] to remove using indexes from last to first
                self.parent.project.removeDatasetIndex(index.row())
                self.parent.statusBar().showMessage('Datasets deleted')


        def _normalize_ds_in_full(self, ds, window=None):
            try:
                ds.normalize_force()
                ds.extract_chi_force()
                ds.extract_ft_force(window=window)
            except:
                pass

        def plot_project_in_E(self):
            if self.list_project.selectedIndexes():
                update_figure([self.figure_project.ax], self.toolbar_project, self.canvas_project)

                for index in self.list_project.selectedIndexes():
                    ds = self.parent.project[index.row()]
                    self._normalize_ds_in_full(ds)
                    # ds.normalize_force()
                    # ds.extract_chi_force()
                    # ds.extract_ft_force()
                    # ds.extract_ft()
                    # ds.extract_ft_force()
                    energy = ds.energy
                    if self.radioButton_mu_xasproject.isChecked():
                        data = ds.mu
                    elif self.radioButton_norm_xasproject.isChecked():
                        if self.checkBox_norm_flat_xasproject.checkState():
                            data = ds.flat
                        else:
                            data = ds.norm
                    if self.checkBox_deriv.isChecked():
                        if not hasattr(ds, 'mu_deriv'):
                            ds.deriv()
                        data = ds.mu_deriv
                        energy = ds.energy_deriv


                    self.figure_project.ax.plot(energy, data, label=ds.name)

                    if self.radioButton_mu_xasproject.isChecked() and not self.checkBox_deriv.isChecked():
                        if self.checkBox_preedge_show.checkState():
                            self.figure_project.ax.plot(ds.energy, ds.pre_edge, label='Preedge', linewidth=0.75)
                        if self.checkBox_postedge_show.checkState():
                            self.figure_project.ax.plot(ds.energy, ds.post_edge, label='Postedge', linewidth=0.75)
                        if self.checkBox_background_show.checkState():
                            self.figure_project.ax.plot(ds.energy, ds.bkg, label='Background', linewidth=0.75)

                self.parent.set_figure(self.figure_project.ax, self.canvas_project, label_x='Energy /eV',
                                label_y=r'$\chi  \mu$' + '(E)'),

                if self.checkBox_force_range_E.checkState():
                    self.figure_project.ax.set_xlim(
                        (float(self.lineEdit_e0.text()) + float(self.lineEdit_range_E_lo.text())),
                        (float(self.lineEdit_e0.text()) + float(self.lineEdit_range_E_hi.text())))
                self.current_plot_in = 'e'

        def plot_project_in_K(self):
            if self.list_project.selectedIndexes():
                update_figure([self.figure_project.ax], self.toolbar_project, self.canvas_project)
                window = self.set_ft_window()
                for index in self.list_project.selectedIndexes():
                    ds = self.parent.project[index.row()]
                    self._normalize_ds_in_full(ds, window=window)

                    data = ds.chi * np.power(ds.k, self.spinBox_k_weight.value())

                    self.figure_project.ax.plot(ds.k, data, label=ds.name)
                    data_max = data.max()
                    if self.checkBox_show_window.isChecked():
                        self.figure_project.ax.plot(ds.k, ds.kwin * data_max / 2, label='Windows')

                self.parent.set_figure(self.figure_project.ax, self.canvas_project,
                                label_x='k (' + r'$\AA$' + '$^1$' + ')',
                                label_y=r'$\chi  \mu$' + '(k)')

                if self.checkBox_force_range_k.checkState():
                    self.figure_project.ax.set_xlim(float(self.lineEdit_range_k_lo.text()),
                                                      float(self.lineEdit_range_k_hi.text()))
                self.current_plot_in = 'k'

        def plot_project_in_R(self):
            if self.list_project.selectedIndexes():
                update_figure([self.figure_project.ax], self.toolbar_project, self.canvas_project)
                window = self.set_ft_window()
                for index in self.list_project.selectedIndexes():
                    ds = self.parent.project[index.row()]
                    # ds.extract_ft_force(window=window)
                    self._normalize_ds_in_full(ds, window=window)
                    if self.checkBox_show_chir_mag.checkState():
                        self.figure_project.ax.plot(ds.r, ds.chir_mag, label=ds.name)
                    if self.checkBox_show_chir_im.checkState():
                        self.figure_project.ax.plot(ds.r, ds.chir_im, label=(ds.name + ' Im'))
                    if self.checkBox_show_chir_re.checkState():
                        self.figure_project.ax.plot(ds.r, ds.chir_re, label=(ds.name + ' Re'))
                    # if self.checkBox_show_chir_pha.checked:
                    #    self.figure_project.ax.plot(ds.r, ds.chir_pha, label=(ds.name + ' Ph'))

                self.parent.set_figure(self.figure_project.ax, self.canvas_project, label_y=r'$\chi  \mu$' + '(k)',
                                label_x='R (' + r'$\AA$' + ')')
                if self.checkBox_force_range_R.checkState():
                    self.figure_project.ax.set_xlim(float(self.lineEdit_range_R_lo.text()),
                                                      float(self.lineEdit_range_R_hi.text()))
                self.current_plot_in = 'R'

        def save_xas_project(self):
            options = QtWidgets.QFileDialog.DontUseNativeDialog
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save XAS project as', self.parent.widget_data.working_folder,
                                                                'XAS project files (*.xas)', options=options)
            if filename:
                if Path(filename).suffix != '.xas':
                    filename = filename + '.xas'
                print(filename)
                self.parent.project.save(filename=filename)

        def open_xas_project(self):
            options = QtWidgets.QFileDialog.DontUseNativeDialog
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load XAS project', self.parent.widget_data.working_folder,
                                                                'XAS project files (*.xas)', options=options)
            if filename:
                self.parent.project_loaded_from_file = xasproject.XASProject()
                self.parent.project_loaded_from_file.load(filename=filename)

                if ret == 0:
                    self.parent.project = self.parent.xasproject_loaded_from_file
                    self.update_project_list(self.parent.project._datasets)
                if ret == 1:
                    for i in self.parent.project_loaded_from_file._datasets:
                        self.parent.project.append(i)

        def save_datasets_as_text(self, send_to_dropbox=False):

            selection = self.list_project.selectedIndexes()
            if selection != []:
                ret = self.message_box_save_datasets_as()
                options = QtWidgets.QFileDialog.DontUseNativeDialog
                # if not send_to_dropbox:
                pathname = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose folder...',
                                                                  self.parent.widget_data.working_folder,
                                                                  options=options)
                # else:
                #     pathname = f'{expanduser("~")}/tmp'
                separator = '#______________________________________________________\n'
                if pathname is not '':
                    for indx, obj in enumerate(selection):
                        ds = self.parent.project._datasets[selection[indx].row()]
                        filename = ds.name
                        if ret == 0:
                            xx = ds.energy
                            yy = np.array(ds.mu)
                            keys = '# energy(eV), mu(E)\n'
                        elif ret == 1:
                            xx = ds.energy
                            yy = ds.norm
                            keys = '# energy(eV), normalized mu(E)\n'
                        elif ret == 2:
                            xx = ds.energy
                            yy = ds.flat
                            keys = '# energy(eV), flattened normalized mu(E)\n'
                        table = np.stack((xx, yy)).T

                        filename_new = '{}/{}.{}'.format(pathname, filename, 'mu')
                        fid = open(filename_new, 'w')
                        header_wo_cols_names = ds.header[0:ds.header.rfind('#')]
                        fid.write(header_wo_cols_names)
                        fid.write(separator)
                        fid.write(keys)
                        fid.close()

                        fid = open(filename_new, 'a')
                        np.savetxt(fid, table)
                        fid.close()

                        # if hasattr(ds, 'ext_data') and (ds.ext_data is not None):
                        #     if 'pil100k_image' in ds.ext_data.keys():
                        #         tiff_files = dump_tiff_images(filename_new, None, ds.ext_data, df_red=ds.df.fillna(0), zip=True)
                        #         if send_to_dropbox:
                        #             if tiff_files is not None:
                        #                 for tiff_file in tiff_files:
                        #                     self.cloud_dispatcher.load_to_dropbox(tiff_file,
                        #                                                           year=ds.md['year'],
                        #                                                           cycle=ds.md['cycle'],
                        #                                                           proposal=ds.md['proposal'])
                        #
                        # if send_to_dropbox:
                        #     self.cloud_dispatcher.load_to_dropbox(filename_new,
                        #                                           year = ds.md['year'],
                        #                                           cycle = ds.md['cycle'],
                        #                                           proposal = ds.md['proposal'])


        def _intersect_metadata_dicts(self, md_list):

            keys1 = [k for k in md_list[0].keys()]
            values1 = [md_list[0][k] for k in keys1]
            keys_intersected = []
            for k in keys1:
                this_key_cond = True
                for each_md in md_list:
                    if each_md[k] not in values1:
                        this_key_cond = False
                if this_key_cond:
                    keys_intersected.append(k)
            md_common = {k : md_list[0][k] for k in md_list[0].keys() if k in keys_intersected}
            return md_common


        def merge_datasets(self):
            selection = self.list_project.selectedIndexes()
            if selection != []:
                # here we check that what teh limits are
                energy_min = []
                energy_max = []
                for indx, obj in enumerate(selection):
                    _ds = self.parent.project._datasets[selection[indx].row()]
                    energy_min.append(_ds.energy[0])
                    energy_max.append(_ds.energy[-1])
                #print(energy_min)
                #print(energy_max)
                energy_min_median = np.median(energy_min)
                energy_max_median = np.median(energy_max)

                ext_data_list = []

                energy_range = np.array(energy_max) -np.array(energy_min)
                #print(energy_range)
                master_index = np.argmax(energy_range)
                #print(master_index)
                mu = self.parent.project._datasets[selection[master_index].row()].mu
                energy_master = self.parent.project._datasets[selection[master_index].row()].energy
                mu_array = np.zeros([len(selection), len(mu)])


                merged_files_string = ['# merged files\n']

                merged_md_list = []
                name_list = []
                for indx, obj in enumerate(selection):
                    _ds = self.parent.project._datasets[selection[indx].row()]
                    energy = _ds.energy
                    #if (np.abs(energy[0]-energy_min_median)<10) and (np.abs(energy[-1]-energy_max_median)<10):
                    # mu = self.parent.project._datasets[selection[indx].row()].mu.mu
                    mu = _ds.mu
                    mu = np.interp(energy_master, energy, mu)
                    mu_array[indx, :] = mu

                    # _df['mut'][indx, :] = _ds.df['mut']
                    # _df['muf'][indx, :] = _ds.df['muf']
                    # _df['mur'][indx, :] = _ds.df['mur']

                    merged_md_list.append(_ds.md)
                    merged_files_string.append('# ' + _ds.filename + '\n')
                    name_list.append(_ds.name)
                    if hasattr(_ds, 'ext_data'):
                        ext_data_list.append(_ds.ext_data)
                    # this_uid = _ds.md['uid']
                    #     # self.parent.project._datasets[selection[indx].row()].md['uid']
                    # # merged_uids_string.append('# ' + this_uid + '\n')
                    # merged_uids_string_for_md.append(this_uid)

                merged_name = os.path.commonprefix(name_list) + ' merged'
                # mask = np.all(mu_array !=0, axis=1)
                # mu_array = mu_array[mask]
                np.savetxt('/nsls2/data3/iss/legacy/Sandbox/data.dat', mu_array)
                self.merge_mu=mu_array

                #evaluate zscore
                _zscores = zscore(self.merge_mu, axis=1)
                zscores=np.average(_zscores, axis=1)
                self.zscores = zscores
                print(zscores)
                self.merge_energy =  energy_master
                mu_merged = np.average(mu_array, axis=0)

                # df = pd.DataFrame({'energy' : energy_master,
                #                    'mut' : np.average(_df['mut'], axis=0),
                #                    'muf' : np.average(_df['muf'], axis=0),
                #                    'mur' : np.average(_df['mur'], axis=0)})

                if len(ext_data_list) > 0:
                    ext_data_merged = copy.deepcopy(ext_data_list[0])
                    for i in range(1, len(ext_data_list)):
                        for k in ext_data_merged.keys():
                            if k != 'data_kind':
                                if type(ext_data_merged[k]) == dict:
                                    for sub_k in ext_data_merged[k].keys():
                                        ext_data_merged[k][sub_k] += ext_data_list[i][k][sub_k]
                                else:
                                    ext_data_merged[k] += ext_data_list[i][k]

                    for k in ext_data_merged.keys():
                        if k != 'data_kind':
                            if type(ext_data_merged[k]) == dict:
                                for sub_k in ext_data_merged[k].keys():
                                    ext_data_merged[k][sub_k] = ext_data_merged[k][sub_k]/len(ext_data_list)
                            else:
                                ext_data_merged[k] /= len(ext_data_list)
                else:
                    ext_data_merged = None

                merged = XASDataSet(name=merged_name, md=merged_files_string, energy=energy_master, mu=mu_merged, filename='',
                                               datatype='processed', ext_data=ext_data_merged)#, df=df)
                merged.header = "".join(merged.md)

                merged.md = self._intersect_metadata_dicts(merged_md_list)
                merged.md['merged files'] = "".join(merged_files_string)
                # merged.md['merged uids'] = "".join(merged_uids_string)
                #merged.md['uid'] = str(merged_uids_string_for_md)
                self.parent.project.append(merged)
                self.parent.project.project_changed()

        def combine_and_save_datasets_as_text(self):
            selection = self.list_project.selectedIndexes()
            if selection != []:
                ds_list = []
                md = []
                for indx, obj in enumerate(selection):
                    ds_list.append(self.parent.project._datasets[selection[indx].row()])

                ds_list.sort(key=lambda x: x.name)
                mu = ds_list[0].mu
                mu_array = np.zeros([len(selection) + 1, len(mu)])
                energy_master = ds_list[0].energy

                mu_array[0, :] = energy_master
                ret = self.message_box_save_datasets_as()
                for indx, obj in enumerate(selection):
                    ds = ds_list[indx]
                    energy = ds.energy
                    if ret == 0:
                        yy = np.array(ds.mu)
                        keys = '# energy(eV), mu(E)\n'
                    elif ret == 1:
                        yy = ds.norm
                        keys = '# energy(eV), normalized mu(E)\n'
                    elif ret == 2:
                        yy = ds.flat
                        keys = '# energy(eV), flattened normalized mu(E)\n'

                    yy = np.interp(energy_master, energy, yy)
                    mu_array[indx + 1, :] = yy
                    md.append(ds.name)

                self.mu_array = mu_array
                options = QtWidgets.QFileDialog.DontUseNativeDialog
                filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save XAS project', self.parent.widget_data.working_folder,
                                                                    'XAS dataset (*.dat)', options=options)
                if filename:
                    if Path(filename).suffix != '.xas':
                        filename = filename + '.xas'
                    print(filename)
                    filelist = "{}".format("\n".join(md[0:]))
                    separator = '\n #______________________________________________________\n'

                    header = '{} {} {}'.format(filelist, separator, keys)
                    fid = open(filename, 'w')
                    np.savetxt(fid, np.transpose(mu_array), header=header)
                    fid.close()

        def rename_dataset(self):
            selection = self.list_project.selectedIndexes()
            if selection != []:
                name = self.parent.project._datasets[selection[0].row()].name
                new_name, ok = QtWidgets.QInputDialog.getText(self, 'Rename dataset', 'Enter new name:',
                                                              QtWidgets.QLineEdit.Normal, name)
                if ok:
                    self.parent.project._datasets[selection[0].row()].name = new_name
                    self.parent.project.project_changed()

        # def save_datasets_to_database(self):
        #     selection = self.list_project.selectedIndexes()
        #     if selection != []:
        #
        #         for indx, _ in enumerate(selection):
        #             ds = self.parent.project._datasets[selection[indx].row()]
        #             sample_name = ds.name
        #             compound = ds.name
        #             try:
        #                 element = ds.md['element']
        #                 edge = ds.md['edge']
        #                 uid = ds.md['uid']
        #             except:
        #                 element = ''
        #                 edge = ''
        #                 uid = ''
        #
        #             e0 = ds.e0
        #             reference = 0
        #             self._dlg = MetadataDialog(sample_name, compound, element, edge, e0, reference, uid, parent=self)
        #             if self._dlg.exec_():
        #                 sample_name, compound, element, edge, e0, reference, uid = self._dlg.getValues()
        #                 metadata = {'Sample_name': sample_name,
        #                             'compound': compound,
        #                             'Element' : element,
        #                             'Edge' : edge,
        #                             'E0': e0,
        #                             'Reference' : reference,
        #                             'ISS_DB_uid' : uid}
        #                 try:
        #                     mu_norm = ds.flat.values
        #                 except AttributeError:
        #                     mu_norm = ds.flat
        #                 energy = ds.energy
        #                 data = {'Energy': energy, 'mu_norm': mu_norm}
        #                 save_spectrum_to_db(metadata, data)


        def export_dataset_to_mcr_project(self):
            selection = self.list_project.selectedIndexes()
            if selection != []:
                index = [i.row() for i in selection]

                name, ok = QtWidgets.QInputDialog.getText(self, 'Dataset name', 'Enter name:',
                                                              QtWidgets.QLineEdit.Normal, 'New Dataset')
                # TODO: add metadata to the output
                # TODO: turn t into time
                if ok:
                    energy, t_dict, data = self.parent.project.convert_into_2d_dataset(np.sort(index))

                    self.parent.widget_mcr._create_dataset(energy, t_dict, data, name=name)

        def export_ref_to_mcr_project(self):
            selection = self.list_project.selectedIndexes()
            if selection != []:
                x_list, data_list, label_list = [], [], []
                for i in selection:
                    ds = self.parent.project[i.row()]
                    x_list.append(ds.energy)
                    data_list.append(ds.flat)
                    label_list.append(ds.name)

                self.parent.widget_mcr.add_references_to_specific_set(x_list, data_list, label_list)

                # name, ok = QtWidgets.QInputDialog.getText(self, 'Dataset name', 'Enter name:',
                #                                           QtWidgets.QLineEdit.Normal, 'New Dataset')
                # # TODO: add metadata to the output
                # # TODO: turn t into time
                # if ok:
                #     energy, t_dict, data = self.parent.project.convert_into_2d_dataset(np.sort(index))
                #
                #     self.parent.widget_mcr._create_dataset(energy, t_dict, data, name=name)







        def truncate(self):
            sender = QObject()
            sender_object = sender.sender().objectName()
            print(sender_object)
            selection = self.list_project.selectedIndexes()
            if selection != []:
                for indx, obj in enumerate(selection):
                    print(indx)
                    ds = self.parent.project._datasets[selection[indx].row()]
                    print(ds.name)
                    energy = ds.energy
                    mu = ds.mu
                    indx_energy_to_truncate_at = (np.abs(energy - float(self.lineEdit_truncate_at.text()))).argmin()

                    if sender_object == 'pushButton_truncate_below':
                        ds.energy = energy[indx_energy_to_truncate_at:]
                        ds.mu = mu[indx_energy_to_truncate_at:]

                    elif sender_object == 'pushButton_truncate_above':
                        ds.energy = energy[0:indx_energy_to_truncate_at]
                        ds.mu = mu[0:indx_energy_to_truncate_at]
                    ds.update_larch()
                    self.parent.project._datasets[selection[indx].row()] = ds

        '''
         Service routines
        '''

        def message_box_save_datasets_as(self):
            messageBox = QtWidgets.QMessageBox()
            messageBox.setText('Save datasets as..')
            messageBox.addButton(QtWidgets.QPushButton('mu(E)'), QtWidgets.QMessageBox.YesRole)
            messageBox.addButton(QtWidgets.QPushButton('normalized mu(E)'), QtWidgets.QMessageBox.NoRole)
            messageBox.addButton(QtWidgets.QPushButton('flattened mu(E)'), QtWidgets.QMessageBox.NoRole)
            ret = messageBox.exec_()
            return ret

        def message_box_warning(self, line1='Warning', line2=''):

            messageBox = QtWidgets.QMessageBox()
            messageBox.setText(line1)
            if line2:
                messageBox.setInformativeText(line2)
            messageBox.setWindowTitle("Warning")
            messageBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            messageBox.exec_()

        def set_ft_window(self):
            window = dict()
            window['window_type'] = self.windows_list[self.comboBox_window.currentIndex()]
            window['r_weight'] = self.spinBox_r_weight.value()
            try:
                window['tapering'] = float(self.lineEdit_window_tapering.text())
            except:
                window['tapering'] = 1

            return window


        def plot_project_in_E_norm_by_max(self):
            if self.list_project.selectedIndexes():
                update_figure([self.figure_project.ax], self.toolbar_project, self.canvas_project)

                for index in self.list_project.selectedIndexes():
                    ds = self.parent.project[index.row()]
                    # self._normalize_ds_in_full(ds)
                    # ds.normalize_force()
                    # ds.extract_chi_force()
                    # ds.extract_ft_force()
                    # ds.extract_ft()
                    # ds.extract_ft_force()
                    energy = ds.energy
                    # if self.radioButton_mu_xasproject.isChecked():
                    data = ds.mu.copy()
                    data -= data[0]
                    data /= data.max()
                    # elif self.radioButton_norm_xasproject.isChecked():
                    #     if self.checkBox_norm_flat_xasproject.checkState():
                    #         data = ds.flat
                    #     else:
                    #         data = ds.norm
                    # if self.checkBox_deriv.isChecked():
                    #     if not hasattr(ds, 'mu_deriv'):
                    #         ds.deriv()
                    #     data = ds.mu_deriv
                    #     energy = ds.energy_deriv


                    self.figure_project.ax.plot(energy, data, label=ds.name)

                    # if self.radioButton_mu_xasproject.isChecked() and not self.checkBox_deriv.isChecked():
                    #     if self.checkBox_preedge_show.checkState():
                    #         self.figure_project.ax.plot(ds.energy, ds.pre_edge, label='Preedge', linewidth=0.75)
                    #     if self.checkBox_postedge_show.checkState():
                    #         self.figure_project.ax.plot(ds.energy, ds.post_edge, label='Postedge', linewidth=0.75)
                    #     if self.checkBox_background_show.checkState():
                    #         self.figure_project.ax.plot(ds.energy, ds.bkg, label='Background', linewidth=0.75)

                self.parent.set_figure(self.figure_project.ax, self.canvas_project, label_x='Energy /eV',
                                label_y=r'$\chi  \mu$' + '(E)'),

                # if self.checkBox_force_range_E.checkState():
                #     self.figure_project.ax.set_xlim(
                #         (float(self.lineEdit_e0.text()) + float(self.lineEdit_range_E_lo.text())),
                #         (float(self.lineEdit_e0.text()) + float(self.lineEdit_range_E_hi.text())))
                self.current_plot_in = 'e'

        def show_ext_data(self):
            selection = self.list_project.selectedIndexes()
            if selection != []:
                indices = [i.row() for i in selection]
                for index in indices:
                    ds = self.parent.project[index]
                    ext_data = ds.ext_data
                    if ext_data['data_kind'] == b'von_hamos':
                        plt.figure(1, clear=True)
                        plt.subplot(221)
                        plt.contourf(ext_data['pil100k_roi1_vh']['pixel'], ds.energy, ext_data['pil100k_roi1_vh']['intensity'], 51)



########




