from PyQt5.QtWidgets import QApplication
import pyqt_viewer
import argparse


def main(args):
    qapp = QApplication([])
    main_window = pyqt_viewer.MainWindow(input_path=args.input)

    main_window.setWindowTitle("SMAL Model Viewer")
    main_window.show()
    qapp.exec_()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None, type=str)
    args = parser.parse_args()

    main(args)
