try:
    from PySide6.QtWidgets import QApplication
    from gui.main_window import MainWindow  # Import the custom MainWindow
except ImportError:
    raise ImportError("To use the GUI features, please install the package PySide6.\n"
                      "You can install it using the command:\n"
                      "pip install PySide6\n")

def launch_gui():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()

def main():
    launch_gui()

if __name__ == "__main__":
    main()
