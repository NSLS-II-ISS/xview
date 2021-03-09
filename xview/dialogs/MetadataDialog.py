from PyQt5 import uic, QtGui, QtCore
import pkg_resources

ui_path = pkg_resources.resource_filename('xview', 'dialogs/MetadataDialog.ui')

class MetadataDialog(*uic.loadUiType(ui_path)):

    def __init__(self, sample_name, compound, element, edge, e0, reference, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.setWindowTitle('Metadata for processed DB')
        self.lineEdit_sample_name.setText(f'{sample_name}')
        self.lineEdit_compound.setText(f'{compound}')
        self.lineEdit_element.setText(f'{element}')
        self.lineEdit_edge.setText(f'{edge}')
        self.lineEdit_e0.setText(f'{e0}')
        self.checkBox_reference.setCheckState(reference)


    def getValues(self):
        referenceFlag = int(self.checkBox_reference.checkState() == 2)
        return (self.lineEdit_sample_name.text(), self.lineEdit_compound.text(),
                self.lineEdit_element.text(), self.lineEdit_edge.text(), self.lineEdit_e0.text(),
                referenceFlag)
