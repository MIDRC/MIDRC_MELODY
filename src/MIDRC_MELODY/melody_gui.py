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

from importlib import resources as _resources
import sys
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QIcon, QPixmap
    from MIDRC_MELODY.gui.main_window import MainWindow  # Import the custom MainWindow
except ImportError as e:
    raise ImportError("To use the GUI features, please install the package PySide6.\n"
                      "You can install it using the command:\n"
                      "pip install PySide6\n"
                      "\n"
                      "Error details: " + str(e))


# Global list to hold window references.
_open_windows = []

def _load_package_icon() -> QIcon:
    """
    Load the bundled icon from the installed package resources.
    Uses importlib.resources.files(...) with the current package name.
    """
    try:
        pkg = __package__ or "MIDRC_MELODY"
        data = _resources.files(pkg).joinpath("resources", "MIDRC.ico").read_bytes()
        pix = QPixmap()
        pix.loadFromData(data)
        return QIcon(pix)
    except Exception:
        return QIcon()

# optional: import ctypes only on Windows
_ctypes = None
if sys.platform == "win32":
    try:
        import ctypes as _ctypes
    except Exception:
        _ctypes = None

def _set_windows_appid(appid: str) -> None:
    """
    On Windows set an explicit AppUserModelID so the taskbar uses the correct icon.
    Must be called before creating any windows / QApplication.
    """
    if sys.platform != "win32" or _ctypes is None:
        return
    try:
        _ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)
    except Exception:
        # best-effort; ignore failures
        pass

def launch_gui() -> None:
    _set_windows_appid("com.midrc.melody")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        new_app = True
    else:
        new_app = False

    icon = _load_package_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)

    window = MainWindow()
    if not icon.isNull():
        window.setWindowIcon(icon)
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
