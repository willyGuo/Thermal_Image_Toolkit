"""app."""

import sys
from thermal_toolkit.core.qt import QApplication
from thermal_toolkit.ui.main_window import ThermalRawToolkit


def main():
    app=QApplication(sys.argv)
    app.setStyle("Fusion")
    w=ThermalRawToolkit(); w.showMaximized()
    sys.exit(app.exec_())
