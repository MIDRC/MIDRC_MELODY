from __future__ import annotations
import os
import sys
import re
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Optional, Match, Final

from PySide6.QtCore import Slot, QSettings, QRunnable, QThreadPool, QObject, Signal, QCoreApplication
from PySide6.QtGui import QTextCursor, QFontDatabase, QFont, QTextCharFormat
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


"""ANSI‑aware redirect stream for QPlainTextEdit.

    • Handles tqdm’s carriage‑return / cursor‑up logic so stacked bars overwrite
      instead of appending.
    • Skips the *newline that terminates the bar* while it is still in‑flight,
      but lets the final newline through when the bar closes.
"""
__all__ = ["ANSIProcessor"]

_CSI_RE : Final = re.compile(r"\x1b\[(\d*)([A-Za-z])")

def _next_csi_is_cursor_up(buf: str, pos: int) -> bool:
    """Return True when buf[pos:] starts with ESC[…A (cursor‑up)."""
    if pos >= len(buf) or buf[pos] != "\x1b":
        return False
    m = _CSI_RE.match(buf, pos)
    return bool(m and m.group(2) == "A")


class ANSIProcessor:
    @staticmethod
    def process(console: QPlainTextEdit, chunk: str) -> None:
        # one persistent flag per widget
        if not hasattr(console, "_nl_pending"):
            console._nl_pending = False          # type: ignore[attr-defined]

        cursor = console.textCursor()
        i, n = 0, len(chunk)

        while i < n:
            # ── commit any deferred newline *before* inspecting next byte
            if console._nl_pending:
                if not _next_csi_is_cursor_up(chunk, i):
                    cursor.insertBlock()         # real newline
                console._nl_pending = False

            ch = chunk[i]

            # ── carriage return  → clear line
            if ch == "\r":
                cursor.movePosition(QTextCursor.StartOfLine)
                cursor.select(QTextCursor.LineUnderCursor)
                cursor.removeSelectedText()
                i += 1
                continue

            # ── line‑feed  → decide later
            if ch == "\n":
                # Try to step to the next existing row; if we’re already on
                # the last row of the document, create a new one once.
                if not cursor.movePosition(QTextCursor.Down):
                    cursor.insertBlock()  # adds a row only if needed
                cursor.movePosition(QTextCursor.StartOfLine)  # like real terminal ▼
                i += 1
                continue

            # ── CSI escape sequences
            if ch == "\x1b":
                m = _CSI_RE.match(chunk, i)
                if m:
                    num = int(m.group(1) or 1)
                    cmd = m.group(2)

                    if cmd == "A":              # cursor‑up
                        for _ in range(num):
                            cursor.movePosition(QTextCursor.Up)
                        cursor.movePosition(QTextCursor.StartOfLine)
                        i = m.end()
                        continue

                    if cmd == "K":              # erase line
                        cursor.select(QTextCursor.LineUnderCursor)
                        cursor.removeSelectedText()
                        cursor.movePosition(QTextCursor.StartOfLine)
                        i = m.end()
                        continue

                    if cmd == "m":  # bold / reset / colours …
                        # keep a per‑widget flag so style persists across chunks
                        if not hasattr(console, "_ansi_fmt"):
                            console._ansi_fmt = QTextCharFormat()  # type: ignore[attr-defined]

                        # SGR can have multiple semicolon‑separated params
                        for p in (m.group(1) or "0").split(";"):
                            p = p or "0"
                            if p == "0":  # reset all
                                console._ansi_fmt = QTextCharFormat()
                            elif p == "1":  # bold on
                                console._ansi_fmt.setFontWeight(QFont.Bold)
                            elif p == "22":  # normal intensity
                                console._ansi_fmt.setFontWeight(QFont.Normal)
                            # you can extend here for colours (30–37 / 90–97, etc.)

                        cursor.setCharFormat(console._ansi_fmt)
                        i = m.end()
                        continue
                # any other ESC sequence is ignored

            # ── printable character
            cursor.insertText(ch)
            i += 1

        console.setTextCursor(cursor)
        console.ensureCursorVisible()


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

