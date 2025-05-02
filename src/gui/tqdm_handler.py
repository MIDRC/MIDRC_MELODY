import os
import sys
import re
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Optional, Match

from PySide6.QtCore import Slot, QSettings, QRunnable, QThreadPool, QObject, Signal
from PySide6.QtGui import QTextCursor, QFontDatabase, QFont
from PySide6.QtWidgets import QPlainTextEdit

# ──────────────────────────────────────────────────────────────────────────────
# tqdm‑to‑Qt bridge
# ──────────────────────────────────────────────────────────────────────────────
class EmittingStream(QObject):
    """File‑like object that forwards its *raw* text to Qt."""

    textWritten: Signal = Signal(str)

    def write(self, data: str) -> None:  # noqa: D401 (simple verb)
        data = str(data)  # Ensure that the incoming data is a string
        if data:
            self.textWritten.emit(data)

    def flush(self) -> None:  # noqa: D401 (simple verb)
        pass

    # convince tqdm we are an interactive TTY so it keeps fancy output
    def isatty(self) -> bool:  # noqa: D401 (simple verb)
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Minimal ANSI interpreter
# ──────────────────────────────────────────────────────────────────────────────
class ANSIProcessor:
    """Handle just the escape sequences tqdm uses for stacked progress‑bars."""

    ANSI_RE = re.compile(r"\x1b\[(?P<num>\d*?)(?P<cmd>[A-Za-z])")

    def __init__(self, console: QPlainTextEdit) -> None:  # store widget ref
        self.console = console

    # ---------------------------------------------------------------------
    # public API – feed new bytes; we update the widget in place
    # ---------------------------------------------------------------------
    def feed(self, chunk: str) -> None:  # noqa: C901 (single responsibility)
        """Process *chunk* (plain + escape codes) and mutate *console*."""

        cursor = self.console.textCursor()
        i = 0
        while i < len(chunk):
            ch = chunk[i]

            # (1) Carriage return – rewrite current line
            if ch == "\r":
                cursor.movePosition(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
                cursor.removeSelectedText()
                i += 1
                continue

            # (2) Escape sequence
            if ch == "\x1b":
                match: Optional[Match[str]] = self.ANSI_RE.match(chunk, i)
                if match:
                    cmd = match["cmd"]
                    num = int(match["num"] or 1)
                    if cmd == "A":  # cursor up *num* lines
                        for _ in range(num):
                            cursor.movePosition(QTextCursor.Up)
                        cursor.movePosition(QTextCursor.MoveOperation.StartOfLine)
                    elif cmd == "K" and num in {0, 1, 2}:  # ESC[2K – clear line
                        cursor.movePosition(QTextCursor.MoveOperation.StartOfLine, QTextCursor.KeepAnchor)
                        cursor.removeSelectedText()
                    # ignore other cmds – strip them silently
                    i = match.end()
                    continue
            # (3) Normal character – insert and advance
            cursor.insertText(ch)
            i += 1

        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

# New: Worker signals and Worker class for threaded processing
class WorkerSignals(QObject):
    finished = Signal()
    error = Signal(str)
    result = Signal(object)

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            self.signals.error.emit(str(e))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
