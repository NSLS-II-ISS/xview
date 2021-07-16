import sys
import pkg_resources
from PyQt5 import  QtWidgets, uic
from xview.xasproject.xasproject import XASProject

from xview.widgets import widget_xview_data, widget_xview_project, widget_xview_databroker, widget_xview_rixs

if sys.platform == 'darwin':
    ui_path = pkg_resources.resource_filename('xview', 'ui/ui_xview-mac.ui')
    print('mac')
else:
    ui_path = pkg_resources.resource_filename('xview', 'ui/ui_xview.ui')

class XviewGui(*uic.loadUiType(ui_path)):
    def __init__(self, db=None, db_proc=None, *args, **kwargs):
        self.db = db
        self.db_proc = db_proc
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.project = XASProject()

        self.widget_data = widget_xview_data.UIXviewData(db=db, parent=self)
        self.layout_data.addWidget(self.widget_data)

        self.widget_project = widget_xview_project.UIXviewProject(db_proc=db_proc, parent=self)
        self.layout_project.addWidget(self.widget_project)

        # self.widget_databroker = widget_xview_databroker.UIXviewDatabroker(db=db, parent=self)
        self.widget_databroker = widget_xview_databroker.get_SearchAndOpen_widget(parent=self)
        self.layout_databroker.addWidget(self.widget_databroker)

        self.widget_rixs = widget_xview_rixs.UIXviewRIXS(db=db, parent=self)
        self.layout_rixs.addWidget(self.widget_rixs)

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
