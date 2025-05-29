#  Copyright (c) 2025 Medical Imaging and Data Resource Center (MIDRC).
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.
#

try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer, Qt
    from MIDRC_MELODY.gui.main_window import MainWindow  # Import the custom MainWindow
except ImportError:
    raise ImportError("To use the GUI features, please install the package PySide6.\n"
                      "You can install it using the command:\n"
                      "pip install PySide6\n")


# Global list to hold window references.
_open_windows = []


def launch_gui() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        new_app = True
    else:
        new_app = False
    window = MainWindow()
    _open_windows.clear()
    _open_windows.append(window)
    
    # Store original window flags
    original_flags = window.windowFlags()
    
    window.show()
    # Temporarily add the always-on-top flag without losing other flags.
    QTimer.singleShot(100, lambda: (
         window.setWindowFlags(original_flags | Qt.WindowStaysOnTopHint),
         window.show()
    ))
    # After a short delay, restore the original flags with the window still in the front.
    QTimer.singleShot(300, lambda: (
         window.setWindowFlags(original_flags),
         window.show()
    ))
    if new_app:
        app.exec()


def main():
    launch_gui()


if __name__ == "__main__":
    main()
