from PyQt5 import uic, QtGui, QtCore
from xview.resources import resource_filename
import os

ui_path = resource_filename('xview', 'dialogs/FileMetadataDialog.ui')

class FileMetadataDialog(*uic.loadUiType(ui_path)):

    def __init__(self, path_to_file, header, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.setWindowTitle('File Metadata')
        (path, filename) = os.path.split(path_to_file)

        self.label_folder.setText(f'Folder: {path}')
        self.label_file.setText(f'File: {filename}')

        self.textEdit_metadata.setText(header)


