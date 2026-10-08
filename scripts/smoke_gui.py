"""Open every desktop tab briefly using the real installed ISS packages."""

import os
from tempfile import TemporaryDirectory


def main():
    # Keep the smoke check separate from the user's saved folders and settings.
    with TemporaryDirectory(prefix="xview-smoke-") as directory:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ["XDG_CONFIG_HOME"] = directory
        os.environ["MPLCONFIGDIR"] = directory
        os.environ["LARCHDIR"] = directory

        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication
        from xview.xview import XviewGui

        app = QApplication([])
        window = XviewGui()
        try:
            window.show()
            QTimer.singleShot(250, app.quit)
            app.exec_()
            for name in ("widget_data", "widget_project", "widget_statistics",
                         "widget_rixs", "widget_mcr", "widget_xfit"):
                assert getattr(window, name) is not None, name
            print("XView opened successfully with all six desktop widgets.")
        finally:
            window.widget_rixs.processor.stop()
            window.widget_rixs.thread.quit()
            if not window.widget_rixs.thread.wait(5000):
                raise RuntimeError("RIXS worker did not stop")
            window.close()


if __name__ == "__main__":
    main()
