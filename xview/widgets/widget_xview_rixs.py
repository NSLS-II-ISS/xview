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
from matplotlib import cm
import json

from matplotlib.figure import Figure
from xas.xasproject import XASDataSet
from isstools.elements.figure_update import update_figure
from isstools.dialogs.BasicDialogs import message_box
from xas.file_io import load_binned_df_from_file
import pyqtgraph as pg

from xas.spectrometer import parse_rixs_scan, parse_rixslog_scan
from xas.vonhamos import ProcessingThread, ProcessingWorker, ProcessingTask
from queue import Queue
from PyQt5.QtCore import QObject, pyqtSignal, QThread
if platform == 'darwin':
    ui_path = pkg_resources.resource_filename('xview', 'ui/ui_xview_data-mac.ui')
else:
    ui_path = pkg_resources.resource_filename('xview', 'ui/ui_xview_rixs.ui')

pg.setConfigOption('leftButtonPan', False)


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
        self.pushButton_save_calibration_dict.clicked.connect(self.save_calibration_dict)

        self.comboBox_sort_files_by.addItems(['Time','Name'])
        self.comboBox_sort_files_by.currentIndexChanged.connect((self.get_file_list))

        self.comboBox_data_numerator.currentIndexChanged.connect(self.update_current_numerator)
        self.comboBox_data_denominator.currentIndexChanged.connect(self.update_current_denominator)
        self.comboBox_data_bkg.currentIndexChanged.connect(self.update_current_bkg)

        self.list_data.itemSelectionChanged.connect(self.select_files_to_plot)
        self.pushButton_calibrate_with_roi.clicked.connect(self.start_calibration_with_rois)
        self.pushButton_create_xes_data.clicked.connect(self.create_xes_data)
        # self.push_add_to_project.clicked.connect(self.add_data_to_project)
        self.list_data.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_data.customContextMenuRequested.connect(self.xas_data_context_menu)
        self.init_processing_thread()

        self.list_data.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.addCanvas()
        self.add_pyqtgraph_plots()
        self.keys = []
        self.last_keys = []
        self.current_plot_in = ''
        self.binned_data = []
        self.last_numerator= ''
        self.last_denominator = ''
        self.last_bkg = ''
        self._selected_files = []
        self.results = {}
        self.rois = {'1':None, '2':None, '3':None}
        self.__colors = ['red', 'cyan', 'lime']
        for i in range(1,4):

            getattr(self, f"checkBox_roi{i}").toggled.connect(self.add_rois)
            getattr(self, f"checkBox_roi{i}").setStyleSheet("QCheckBox::checked"
                                                            "{"
                                                            f"background-color : {self.__colors[i-1]}"
                                                            "}"
                                                            )

        # Persistent settings
        self.settings = QSettings('ISS Beamline', 'Xview')
        self.working_folder = self.settings.value('working_folder_rixs', defaultValue='/GPFS/xf08id/User Data', type=str)
        self.session_dict = json.loads(self.settings.value('session_dict', defaultValue="{}"))




        _spin_box_objects = ['spinBox_contourf_n', 'doubleSpinBox_contourf_vmin', 'doubleSpinBox_contourf_vmax']
        for sp_obj in _spin_box_objects:
            getattr(self, sp_obj).editingFinished.connect(self.on_spin_box_changed)

        if self.working_folder != '/GPFS/xf08id/User Data':
            self.label_working_folder.setText(self.working_folder)
            self.label_working_folder.setToolTip(self.working_folder)
            self.get_file_list()

    def xas_data_context_menu(self,QPos):
        menu = QMenu()
        plot_action = menu.addAction("&Plot")
        plot_calibration_action = menu.addAction("&Plot as Calibration")
        plot_merge_action = menu.addAction("&Merge and Plot")
        # add_to_project_action = menu.addAction("&Add to project")
        # merge_action = menu.addAction("&Add to project")
        parentPosition = self.list_data.mapToGlobal(QtCore.QPoint(0, 0))
        menu.move(parentPosition+QPos)
        action = menu.exec_()
        if action == plot_action:
            self.plot_rixs_data()
        if action == plot_calibration_action:
            self.start_calibration()
            # self.start_processing(plot_calibration=True, process_scan_file=False)
        if action == plot_merge_action:
            self.merge_and_plot_total()

            pass
            # self.start_processing(plot_calibration=False, process_scan_file=True)
        # elif action == add_to_project_action:
        #     self.add_data_to_project()

    def enforce_fixed_y_dimension(self, key):
        p1, p2 = self.roi[key].getScene


    def add_pyqtgraph_plots(self):

        self.rixs_plot_area = pg.GraphicsLayoutWidget()
        self.rixs_plot_area.setBackground('w')
        self.layout_plot_rixs.addWidget(self.rixs_plot_area)


        self.rixs_plot_item = self.rixs_plot_area.addPlot()
        self.rixs_plot_item.setLabels(left='Pixel', bottom='Pixel', right='Pixel', top='Pixel')
        self.rixs_plot_item.setAspectLocked(False)



        # self.rixs_view = self.rixs_plot_area.addViewBox(lockAspect=False, invertY=True)
        # self.rixs_view.setBackgroundColor('w')

        cmap = cm.get_cmap('jet')
        lut = (cmap(np.linspace(0, 1, 256))[:, :3] *255).astype(np.ubyte)
        self.rixs_image_item = pg.ImageItem(lut=lut)
        self.rixs_plot_item.addItem(self.rixs_image_item)

        self.rixs_overlay_curve = {}

        for i in ['1', '2', '3']:
            self.rixs_overlay_curve[i] = pg.PlotDataItem(pen=pg.mkPen(color='yellow', width=5),
                                                      symbol='o',
                                                      symbolSize=5,
                                                      antialias=True,)

            self.rixs_plot_item.addItem(self.rixs_overlay_curve[i])

        # self.rixs_plot_item = pg.PlotDataItem(pen=pg.mkPen(color='r', width=2), symbol='o', symbolPen='r', symbolBrush='r', symbolSize=3)
        # self.rixs_view.addItem(self.rixs_plot_item)



        # self.rixs_view = pg.ImageView()
        # self.rixs_zoom_view = pg.ImageView()
        # self.rixs_roi = pg.LineROI([0, 100], [400, 100], width=5)
        #
        # self.layout_plot_rixs.addWidget(self.rixs_view)
        # self.layout_plot_rixs_zoom.addWidget(self.rixs_zoom_view)

        # self.figure_linecut = pg.PlotWidget()
        # self.figure_line = self.figure_linecut.plot([1], [1])
        # self.layout_plot_linecut.addWidget(self.figure_linecut)


    def addCanvas(self):
        # self.figure_rixs = Figure()
        # #self.figure_data.set_facecolor(color='#E2E2E2')
        # self.figure_rixs.ax = self.figure_rixs.add_subplot(111)
        # self.canvas = FigureCanvas(self.figure_rixs)
        # self.toolbar = NavigationToolbar(self.canvas, self)
        # self.toolbar.resize(1, 10)
        # self.layout_plot_rixs.addWidget(self.toolbar)
        # self.layout_plot_rixs.addWidget(self.canvas)
        # self.figure_rixs.tight_layout()
        # self.canvas.draw()


        self.figure_linecut = Figure()
        self.figure_linecut.ax = self.figure_linecut.add_subplot(111)
        self.canvas_linecut = FigureCanvas(self.figure_linecut)
        self.toolbar_linecut = NavigationToolbar(self.canvas_linecut, self)
        self.toolbar_linecut.resize(1, 10)
        self.layout_plot_linecut.addWidget(self.toolbar_linecut)
        self.layout_plot_linecut.addWidget(self.canvas_linecut)
        self.figure_linecut.tight_layout()
        self.canvas_linecut.draw()

    def select_working_folder(self):
        self.working_folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a folder", self.working_folder,
                                                                        QtWidgets.QFileDialog.ShowDirsOnly)
        if self.working_folder:
            self.set_working_folder()

    def set_working_folder(self):
        self.settings.setValue('working_folder_rixs', self.working_folder)
        if len(self.working_folder) > 50:
            self.label_working_folder.setText(self.working_folder[1:20] + '...' + self.working_folder[-30:])
        else:
            self.label_working_folder.setText(self.working_folder)
        self.get_file_list()

    def get_file_list(self):
        if self.working_folder:
            self.list_data.clear()
            try:
                self.file_list = [f for f in os.listdir(self.working_folder) if f.endswith('.h5')]
            except:
                self.file_list = []

            if self.comboBox_sort_files_by.currentText() == 'Name':
                self.file_list.sort()
            elif self.comboBox_sort_files_by.currentText() == 'Time':
                self.file_list.sort(key=lambda x: os.path.getmtime('{}/{}'.format(self.working_folder, x)))

                self.file_list.reverse()
            self.list_data.addItems(self.file_list)

    def get_selected_files(self):
        self._selected_files = []
        for item in self.list_data.selectedIndexes():
            _file = f'{self.working_folder}/{item.data()}'
            self._selected_files.append(_file)
        return self._selected_files


    def start_processing(self, plot_calibration=False, process_scan_file=False, plot_and_merge=False):
        self._selected_files = []

        for item in self.list_data.selectedIndexes():
            _file = f'{self.working_folder}/{item.data()}'
            self._selected_files.append(_file)

        self.file_queue = Queue()
        self.file_queue.put(self._selected_files)


        if self.file_queue.qsize() == 0:
            self.log.appendPlainText("⚠️ No files in queue.\n")
            return

        # Set up thread and worker
        self.thread = QThread()
        self.processor = ProcessingThread(self.file_queue, plot_calibration=plot_calibration, process_scan_file=process_scan_file)
        self.processor.moveToThread(self.thread)

        # Connect signals
        self.thread.started.connect(self.processor.run)
        self.processor.result_ready.connect(self.on_result_ready)
        # self.processor.log_message.connect(self.log.appendPlainText)
        # self.processor.progress.connect(self.progress_bar.setValue)
        self.processor.finished.connect(self.thread.quit)
        self.processor.finished.connect(self.on_processing_finished)

        # Start thread
        self.thread.start()
        # self.btn_start.setEnabled(False)
        # self.log.appendPlainText("⚙️ Processing started...\n")

    def get_data_dict_for_plotting(self, plot_type=None):
        dictionary = {}
        if plot_type == 'total':
            dictionary['image'] = self.results['image_total']
            dictionary['overlay'] = None
            dictionary['linecut'] = None #{'total': {'x': [], 'y': []}}

        if plot_type == 'auto_calibration':
            dictionary['image'] = self.results['data']['processed']['image_total']['image']
            dictionary['overlay'] = {}
            dictionary['overlay']['auto_calibration'] = {}
            dictionary['overlay']['auto_calibration']['x'] = self.results['data']['processed']['pixels']['x_centers']
            dictionary['overlay']['auto_calibration']['y'] = self.results['data']['processed']['pixels']['y_centers']

            dictionary['linecut'] = {}
            dictionary['linecut']['auto_calibration'] = {}
            dictionary['linecut']['auto_calibration']['x'] = self.results['data']['processed']['intensity_total']['x']
            dictionary['linecut']['auto_calibration']['y'] = self.results['data']['processed']['intensity_total']['y']
            dictionary['linecut']['auto_calibration']['fit'] = self.results['data']['processed']['intensity_total']['fit']


        if plot_type == 'roi_calibration':
            dictionary['image'] = self.results['image_total']
            dictionary['overlay'] = {}
            dictionary['linecut'] = {}
            for key in self.results['data'].keys():
                dictionary['overlay'][key] = {}
                dictionary['overlay'][key]['x'] = self.results['data'][key]['processed']['pixels']['x_centers']
                dictionary['overlay'][key]['y'] = self.results['data'][key]['processed']['pixels']['y_centers']

                dictionary['linecut'][key] = {}
                dictionary['linecut'][key]['x'] = self.results['data'][key]['processed']['intensity_total']['x']
                dictionary['linecut'][key]['y'] = self.results['data'][key]['processed']['intensity_total']['y']
                dictionary['linecut'][key]['fit'] = self.results['data'][key]['processed']['intensity_total']['fit']

        return dictionary

    def on_result_ready(self, results):
        print(f"Results is ready.")
        self.results = results

        dictionary = self.get_data_dict_for_plotting(plot_type=self.results['plot_type'])
        self.plot_rixs_image(image=dictionary['image'])
        self.plot_overlay_on_rixs(dictionary=dictionary['overlay'])
        self.plot_linecut_batch(dictionary=dictionary['linecut'])




    def on_processing_finished(self):
        print(f"Processing finished.")
        # self.log.appendPlainText("✅ All files processed.\n")
    #     self.btn_start.setEnabled(True)

    def select_files_to_plot(self):
        current_file = f'{self.working_folder}/{self.list_data.currentItem().text()}'
        self._selected_files.append(current_file)
        # self.load_files(self._selected_files)


    def plot_rixs_image(self, image=None):
        try:
            self.rixs_image_item.setImage(image)
        except Exception as e:
            print(e)

    def plot_overlay_on_rixs(self, dictionary=None):
        if dictionary is not None:
            for key in dictionary.keys():
                if key == 'auto_calibration':
                    self.rixs_overlay_curve['1'].setData(dictionary[key]['y'], dictionary[key]['x'])
                else:
                    self.rixs_overlay_curve[key].setData(dictionary[key]['y'], dictionary[key]['x'])
        else:
            self.rixs_overlay_curve['1'].setData([], [])

    def plot_linecut_batch(self, dictionary=None):
        self.figure_linecut.ax.clear()
        if dictionary is not None:
            try:
                for key in dictionary:
                    self.figure_linecut.ax.plot(dictionary[key]['x'], dictionary[key]['y'], color='k', marker='o')
                    self.figure_linecut.ax.plot(dictionary[key]['x'], dictionary[key]['fit'], color='r')
            except Exception as e:
                print(e)
        else:
            self.figure_linecut.ax.plot([], [], color='k', marker='o')
            pass

        self.figure_linecut.ax.set_xlabel('Energy (eV)')
        self.figure_linecut.ax.set_ylabel('Intensity')
        self.figure_linecut.tight_layout()
        self.canvas_linecut.draw_idle()

        # f = h5py.File(current_file, 'r')
        # uid_herfds = list(f.keys())
        # f.close()
        # hdr = self.db[uid_herfds[0]]
        # path = hdr.start['interp_filename']
        # df, header = load_binned_df_from_file(path)
        #
        # keys = df.keys()
        # refined_keys = []
        # for key in keys:
        #     if not (('timestamp' in key) or ('energy' in key)):
        #         refined_keys.append(key)
        # self.keys = refined_keys
        # if self.keys != self.last_keys:
        #     self.last_keys = self.keys
        #     self.comboBox_data_numerator.clear()
        #     self.comboBox_data_bkg.clear()
        #     self.comboBox_data_denominator.clear()
        #     self.comboBox_data_numerator.insertItems(0, self.keys)
        #     self.comboBox_data_bkg.insertItems(0, self.keys)
        #     self.comboBox_data_denominator.insertItems(0, self.keys)
        #     if self.last_numerator!= '' and self.last_numerator in self.keys:
        #         indx = self.comboBox_data_numerator.findText(self.last_numerator)
        #         self.comboBox_data_numerator.setCurrentIndex(indx)
        #     if self.last_denominator!= '' and self.last_denominator in self.keys:
        #         indx = self.comboBox_data_denominator.findText(self.last_denominator)
        #         self.comboBox_data_denominator.setCurrentIndex(indx)
        #     if self.last_bkg!= '' and self.last_bkg in self.keys:
        #         indx = self.comboBox_data_bkg.findText(self.last_bkg)
        #         self.comboBox_data_bkg.setCurrentIndex(indx)

    def update_current_numerator(self):
        self.last_numerator= self.comboBox_data_numerator.currentText()
        # print(f'Chanhin last num to {self.last_numerator}')

    def update_current_bkg(self):
        self.last_bkg= self.comboBox_data_numerator.currentText()

    def update_current_denominator(self):
        self.last_denominator= self.comboBox_data_denominator.currentText()
        # print(f'I am there {self.last_denominator}')

    def parse_rixs_scan(self, xes_normalization=True):
        selected_items = (self.list_data.selectedItems())
        update_figure([self.figure_rixs.ax], self.toolbar, self.canvas)
        path = f'{self.working_folder}/{selected_items[0].text()}'
        self.rixs_dict = parse_rixslog_scan(self.db, path, xes_normalization=xes_normalization)

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

    def on_spin_box_changed(self):
        # n = self.spinBox_contourf_n.value()
        vmin = self.doubleSpinBox_contourf_vmin.value()
        vman = self.doubleSpinBox_contourf_vmax.value()
        self.plot_object.set_clim(vmin=vmin, vmax=vman)
        # self.plot_object.set_levels(n)
        self.canvas.draw_idle()

    def plot_rixs_auto_calibration(self, image=None, overlay_x=None, overlay_y=None):
        try:
            self.rixs_image_item.setImage(image)
            self.rixs_overlay_curve['1'].setData(overlay_y, overlay_x)
        except Exception as e:
            print(e)









    def plot_rixs_data(self):
        # self.process_rixs_dict()
        # n = self.spinBox_contourf_n.value()
        # vmin = self.doubleSpinBox_contourf_vmin.value()
        # vmax = self.doubleSpinBox_contourf_vmax.value()
        # _plot_data_ = self._plot_data.copy()
        # _plot_data_[_plot_data_ < vmin] = vmin
        # _plot_data_[_plot_data_ > vmax] = vmax

        if self.results['plot_type'] == 'calibration':
            img = self.results['results']['image_total']
            x = self.results['results']['pixels']['x']
            y = self.results['results']['pixels']['y']
            try:
                # self.plot_object = self.figure_rixs.ax.contourf(x, y, img, levels=100, vmin=0, vmax=10)
                self.rixs_image_item.setImage(img)
            except Exception as e:
                print(e)

            x_pix_center = self.results['results']['pixel_centers']['x_pix_centers']
            y_pix_center = self.results['results']['pixel_centers']['y_pix_centers']
            try:
                self.rixs_overlay_curve['1'].setData(y_pix_center, x_pix_center)
            except Exception as e:
                print(e)
            # self.figure_rixs.ax.set_xlabel('Pixel')
            # self.figure_rixs.ax.set_ylabel('Pixel')
            # self.figure_rixs.tight_layout()
            # self.canvas.draw_idle()



        # self.figure_rixs.ax.contourf(self._energy_in, self._energy_out, self._plot_data.T, n, vmin=vmin, vmax=vmax)
        # self.figure_rixs.ax.set_xlabel('Incident energy, eV')
        # self.figure_rixs.ax.set_ylabel('Emission energy, eV')
        # self.figure_rixs.tight_layout()
        # self.canvas.draw_idle()



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
        # self.figure_rixs.tight_layout()
        # self.canvas.draw_idle()


    def plot_linecut(self, x_data=None, y_data=None, y_fit=None):
        self.figure_linecut.ax.clear()
        try:
            self.figure_linecut.ax.plot(x_data, y_data, color='k', marker='o')
            self.figure_linecut.ax.plot(x_data, y_fit, color='r')
        except Exception as e:
            print(e)

        self.figure_linecut.ax.set_xlabel('Energy (eV)')
        self.figure_linecut.ax.set_ylabel('Intensity')
        self.figure_linecut.tight_layout()
        self.canvas_linecut.draw_idle()





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


    def add_rois(self):
        sender = QObject()
        object = sender.sender()
        index = object.objectName()[-1:]

        if object.checkState():
            if not self.rois.get('index'):
                self.rois[index] = pg.LineROI([50*int(index), 10], [50*int(index), 400], width=20, movable=True, resizable=True,  pen=pg.mkPen(color=self.__colors[int(index)-1], width=2))
            label = pg.TextItem(index, color='k', fill=self.__colors[int(index)-1], anchor=(0,0))
            label.setParentItem(self.rois[index])
            self.rixs_plot_item.addItem(self.rois[index])
            self.rois[index].setVisible(True)
        else:
            # if not self.session_dict.get(f'rois/{index}'):
            #     self.session_dict[f'rois/{index}'] = {}
            #     self.session_dict['rois'][index] = self.get_line_roi_geometery(index)
            # else:
            #     self.session_dict['rois'][index] = self.get_line_roi_geometery(index)
            self.rixs_plot_item.removeItem(self.rois[index])
            self.rois[index] = None



    def get_line_roi_geometery(self, roi_index):

        pos = self.rois[roi_index].pos()
        size = self.rois[roi_index].size()
        angle = np.deg2rad(self.rois[roi_index].angle())

        dx = size.x()
        dy = size.y()

        vec = np.array([dx * np.cos(angle), dy * np.sin(angle)])

        p1 = np.array([pos.x(), pos.y()])
        p2 = p1 + vec
        # p1, p2 = self.rois[roi_index].getEndpoints()
        width = self.rois[roi_index].size().y()
        return {'p1': (p1.x(), p1.y()), 'p2': (p2.x(), p2.y()), 'width': width}

    def save_calibration_dict(self):
        self.calibration_dictionary = {}

        _roi_list = ['1', '2', '3']
        _data_key_list = self.results['data'].keys()
        if not set(_roi_list).isdisjoint(set(_data_key_list)):
            for _roi in _data_key_list:
                self.calibration_dictionary[_roi] = self.results['data'][_roi]['processed']['calibration']
        else:
            self.calibration_dictionary['auto'] = self.results['data']['processed']['calibration']



    def get_slice_of_image(self, roi_index, image=None):
        roi_data, (rr, cc) = self.rois[roi_index].getArrayRegion(image, self.rixs_image_item, returnMappedCoords=True)
        mask = np.zeros_like(image, dtype=bool)

        rr_int = np.clip(np.round(rr).astype(int), 0, image.shape[0] - 1)
        cc_int = np.clip(np.round(cc).astype(int), 0, image.shape[1] - 1)

        mask[rr_int, cc_int] = True

        return np.where(mask, image, 0)

    def get_rois_mask(self, image=None):
        rois_mask = {}
        for i in ['1', '2', '3']:
            if self.rois[i] is not None:
                roi_data, (rr, cc) = self.rois[i].getArrayRegion(image, self.rixs_image_item,
                                                                         returnMappedCoords=True)
                mask = np.zeros_like(image, dtype=bool)
                rr_int = np.clip(np.round(rr).astype(int), 0, image.shape[0] - 1)
                cc_int = np.clip(np.round(cc).astype(int), 0, image.shape[1] - 1)
                mask[rr_int, cc_int] = True
                rois_mask[i] = mask
        return rois_mask

    def init_processing_thread(self):
        self.processor = ProcessingWorker()


        self.thread = QThread()
        self.processor.moveToThread(self.thread)
        # self.thread.started.connect(self.start_worker)
        self.thread.started.connect(self.processor.run)

        # self.thread.started.connect(lambda: QtCore.QTimer.singleShot(0, self.processor.run))
        self.processor.result_ready.connect(self.on_result_ready)
        self.processor.finished.connect(self.on_processing_finished)
        self.processor.finished.connect(self.thread.quit)

        self.thread.start()

    def start_calibration(self):
        files = self.get_selected_files()
        task = ProcessingTask(files=files, plot_calibration=True)
        self.processor.add_task(task)

    def start_calibration_with_rois(self):
        files = self.get_selected_files()
        image = self.rixs_image_item.image
        rois_mask = self.get_rois_mask(image)
        data = {}
        data['image'] = image
        data['rois'] = rois_mask
        task = ProcessingTask(data=data, calibration_with_rois=True)
        self.processor.add_task(task)


    def merge_and_plot_total(self):
        files = self.get_selected_files()
        task = ProcessingTask(files=files, merge_and_plot_total=True)
        self.processor.add_task(task)


    def create_xes_data(self):
        files = self.get_selected_files()
        data = {}
        image = self.rixs_image_item.image
        rois_mask = self.get_rois_mask(image)

        data['rois'] = rois_mask
        data['calibration'] = self.calibration_dictionary
        task = ProcessingTask(files=files, data=data, create_xes_data=True)
        self.processor.add_task(task)



















