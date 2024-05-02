import sys
import pkg_resources
import os
from PyQt5 import  QtWidgets, uic
# from xview.xasproject.xasproject import XASProject
from xas.xasproject import XASProject
# from isscloudtools.cloud_dispatcher import CloudDispatcher
# from isscloudtools.initialize import get_dropbox_service
from issfactortools.widgets import widget_main as widget_mcr
from xview.widgets import widget_xview_data, widget_xview_project
    # widget_xview_databroker, \
    # widget_xview_rixs , widget_xview_stats


if sys.platform == 'darwin':
    ui_path = pkg_resources.resource_filename('xview', 'ui/ui_xview-mac.ui')
    print('mac')
else:
    ui_path = pkg_resources.resource_filename('xview', 'ui/ui_xview.ui')

class XviewGui(*uic.loadUiType(ui_path)):
    def __init__(self,
                 db=None,
                 db_proc=None,
                 db_archive_catalog=None,
                 db_catalog=None,
                 *args, **kwargs):

        self.db = db
        # self.db_proc = db_proc
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.project = XASProject()

        # try:
        #     self.dropbox_service = get_dropbox_service()
        # except:
        #     print("Cloud services cannot be connected")
        #     self.dropbox_service = None
        #
        # try:
        #     self.cloud_dispatcher = CloudDispatcher(dropbox_service=self.dropbox_service)
        # except Exception as e:
        #     print(f'Could not initialize cloud dispatcher:\n{e}')
        #     self.cloud_dispatcher = None

        self.widget_data = widget_xview_data.UIXviewData(db=db, parent=self)
        self.layout_data.addWidget(self.widget_data)

        self.widget_project = widget_xview_project.UIXviewProject(db_proc=db_proc,
                                                                  parent=self)
        self.layout_project.addWidget(self.widget_project)

        # self.widget_statistics = widget_xview_stats.UIXviewStats(db_proc=db_proc,
        #                                                           cloud_dispatcher = self.cloud_dispatcher,
        #                                                           parent=self)
        # self.layout_statistics.addWidget(self.widget_statistics)

        # self.widget_databroker = widget_xview_databroker.UIXviewDatabroker(db=db, parent=self)
        # if db_archive_catalog is not None:
        #     self.widget_databroker_archive = widget_xview_databroker.get_SearchAndOpen_widget(parent=self, catalog=db_archive_catalog)
        #     self.layout_databroker_archive.addWidget(self.widget_databroker_archive)
        #
        # if db_catalog is not None:
        #     self.widget_databroker = widget_xview_databroker.get_SearchAndOpen_widget(parent=self, catalog=db_catalog)
        #     self.layout_databroker.addWidget(self.widget_databroker)
        #
        # if db_proc is not None:
        #     self.widget_databroker_proc = widget_xview_databroker.get_SearchAndOpen_widget(parent=self, catalog=db_proc,
        #                                                                                    columns='columns_proc',
        #                                                                                    add_open_button=False,
        #                                                                                    add_mcr_button=True)
        #     self.layout_databroker_proc.addWidget(self.widget_databroker_proc)
        #
        # self.widget_rixs = widget_xview_rixs.UIXviewRIXS(db=db, parent=self)
        # self.layout_rixs.addWidget(self.widget_rixs)
        #
        # self.widget_mcr = widget_mcr.FactorAnalysisGUI(parent=self)
        # self.layout_mcr.addWidget(self.widget_mcr)



    def  set_figure(self, axis, canvas, label_x='', label_y=''):
        axis.legend(fontsize='small')
        axis.grid(alpha=0.4)
        axis.set_ylabel(label_y, size='13')
        axis.set_xlabel(label_x, size='13')
        canvas.draw_idle()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = XviewGui()
    main.show()

    sys.exit(app.exec_())
