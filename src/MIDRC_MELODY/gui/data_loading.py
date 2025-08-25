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

import json
import yaml
import os

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QDialog,
)
from MIDRC_MELODY.gui.config_editor import ConfigEditor


def load_config_dict() -> dict:
    """
    Load config settings from QSettings or fallback to config.yaml in the repo root.
    Post-process numeric columns so that any ".inf"/"inf" strings become float("inf").
    """
    settings = QSettings("MIDRC", "MIDRC-MELODY")
    config_str = settings.value("config", "")
    if config_str:
        config = json.loads(config_str)
    else:
        # Default path: three levels up from this file
        config_path = os.path.join(os.path.dirname(__file__), ".", ".", ".", "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.CLoader)
        settings.setValue("config", json.dumps(config))

    # Convert any ".inf"/"inf" bins into actual float("inf")
    if "numeric_cols" in config:
        for col, d in config["numeric_cols"].items():
            if "bins" in d:
                processed_bins = []
                for b in d["bins"]:
                    if isinstance(b, (int, float)):
                        processed_bins.append(b)
                    else:
                        try:
                            if b.strip() in [".inf", "inf"]:
                                processed_bins.append(float("inf"))
                            else:
                                processed_bins.append(float(b))
                        except Exception:
                            processed_bins.append(b)
                config["numeric_cols"][col]["bins"] = processed_bins

    return config


@staticmethod
def save_config_dict(config: dict) -> None:
    settings = QSettings("MIDRC", "MIDRC-MELODY")
    settings.setValue("config", json.dumps(config))


def load_config_file(parent):
    """
    Attempt to load configuration from QSettings. If none exists, prompt the
    user to pick a YAML/JSON file.  Any errors are shown via QMessageBox.
    """
    try:
        # This will raise if no config is stored in QSettings
        _ = load_config_dict()
        QMessageBox.information(parent, "Config Loaded", "Configuration loaded from QSettings.")
    except Exception:
        # Prompt user to select a file on disk
        file_path, _ = QFileDialog.getOpenFileName(
            parent,
            "Select Config File",
            os.path.expanduser("~"),
            "Config Files (*.yaml *.json);;All Files (*)"
        )
        if file_path:
            try:
                # Read JSON or YAML and then save into QSettings via save_config_dict
                if file_path.endswith(".json"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        config = yaml.load(f, Loader=yaml.CLoader)

                save_config_dict(config)  # persist into QSettings
                QMessageBox.information(parent, "Config Loaded", "Configuration loaded from file.")
            except Exception as e2:
                QMessageBox.critical(parent, "Error", f"Failed to load selected config file:\n{e2}")
        else:
            QMessageBox.critical(parent, "Error", "No config file selected.")


def edit_config_file(parent):
    """
    Open the ConfigEditor on the current config (pulled via load_config_dict).
    If loading fails, ask whether to pick an existing config file or start blank.
    Upon acceptance, persist via save_config_dict.
    """
    try:
        config = load_config_dict()
        editor = ConfigEditor(config, parent=parent)
        if editor.exec() == QDialog.Accepted:
            save_config_dict(config)
    except Exception as e:
        # If load_config_dict() failed, ask user if they want to pick an existing file
        resp = QMessageBox.question(
            parent,
            "Edit Config",
            f"Failed to load config: {e}\n\nWould you like to select an existing config file?\n"
            "Press Yes to select a file; or No to create a blank config.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if resp == QMessageBox.Yes:
            # User chose to pick a file
            file_path, _ = QFileDialog.getOpenFileName(
                parent,
                "Select Config File",
                os.path.expanduser("~"),
                "Config Files (*.yaml *.json);;All Files (*)"
            )
            if file_path:
                try:
                    if file_path.endswith(".json"):
                        with open(file_path, "r", encoding="utf-8") as f:
                            config = json.load(f)
                    else:
                        with open(file_path, "r", encoding="utf-8") as f:
                            config = yaml.load(f, Loader=yaml.CLoader)
                    save_config_dict(config)
                except Exception as e3:
                    QMessageBox.critical(parent, "Error", f"Failed to load selected config file: {e3}")
                    return
            else:
                QMessageBox.critical(parent, "Error", "No config file selected.")
                return
        else:
            # User chose “No” → start with a blank config
            config = {}

        # Finally, open ConfigEditor on whichever config dict we have
        editor = ConfigEditor(config, parent=parent)
        if editor.exec() == QDialog.Accepted:
            save_config_dict(config)
