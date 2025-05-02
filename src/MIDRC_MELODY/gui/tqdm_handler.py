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

import re
from typing import Final

from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtGui import QFont, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit

__all__ = ["ANSIProcessor", "EmittingStream", "Worker"]

# ANSI control sequence regex
_CSI_RE: Final = re.compile(r"\x1b\[(\d*)([A-Za-z])")


def _next_csi_is_cursor_up(buf: str, pos: int) -> bool:
    if pos >= len(buf) or buf[pos] != "\x1b":
        return False
    m = _CSI_RE.match(buf, pos)
    return bool(m and m.group(2) == "A")


class ANSIProcessor:
    @staticmethod
    def process(console: QPlainTextEdit, chunk: str) -> None:
        ANSIProcessor._ensure_console_flags(console)
        cursor = console.textCursor()
        i, n = 0, len(chunk)
        text_buf = []  # buffer plain text to batch-insert

        while i < n:
            if console._nl_pending:
                ANSIProcessor._commit_pending_newline(console, cursor, chunk, i)

            ch = chunk[i]

            if ch == "\r":
                if text_buf:
                    cursor.insertText("".join(text_buf))
                    text_buf.clear()
                ANSIProcessor._handle_carriage_return(cursor)
                i += 1
                continue

            if ch == "\n":
                if text_buf:
                    cursor.insertText("".join(text_buf))
                    text_buf.clear()
                ANSIProcessor._handle_line_feed(cursor)
                i += 1
                continue

            if ch == "\x1b":
                if text_buf:
                    cursor.insertText("".join(text_buf))
                    text_buf.clear()
                new_pos = ANSIProcessor._handle_csi(console, cursor, chunk, i)
                if new_pos is not None:
                    i = new_pos
                    continue

            text_buf.append(ch)
            i += 1

        if text_buf:
            cursor.insertText("".join(text_buf))

        console.setTextCursor(cursor)
        console.ensureCursorVisible()

    @staticmethod
    def _ensure_console_flags(console: QPlainTextEdit) -> None:
        if not hasattr(console, "_nl_pending"):
            console._nl_pending = False  # type: ignore[attr-defined]
        if not hasattr(console, "_ansi_fmt"):
            console._ansi_fmt = QTextCharFormat()  # type: ignore[attr-defined]

    @staticmethod
    def _commit_pending_newline(
        console: QPlainTextEdit, cursor: QTextCursor, buf: str, pos: int
    ) -> None:
        if not _next_csi_is_cursor_up(buf, pos):
            cursor.insertBlock()
        console._nl_pending = False  # type: ignore[attr-defined]

    @staticmethod
    def _handle_carriage_return(cursor: QTextCursor) -> None:
        cursor.movePosition(QTextCursor.StartOfLine)
        cursor.select(QTextCursor.LineUnderCursor)
        cursor.removeSelectedText()

    @staticmethod
    def _handle_line_feed(cursor: QTextCursor) -> None:
        if not cursor.movePosition(QTextCursor.Down):
            cursor.insertBlock()
        cursor.movePosition(QTextCursor.StartOfLine)

    @staticmethod
    def _handle_csi(
        console: QPlainTextEdit,
        cursor: QTextCursor,
        buf: str,
        pos: int,
    ) -> int | None:
        m = _CSI_RE.match(buf, pos)
        if not m:
            return None
        num = int(m.group(1) or 1)
        cmd = m.group(2)

        if cmd == "A":  # cursor-up
            for _ in range(num):
                cursor.movePosition(QTextCursor.Up)
            cursor.movePosition(QTextCursor.StartOfLine)
            return m.end()

        if cmd == "K":  # erase line
            cursor.select(QTextCursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.movePosition(QTextCursor.StartOfLine)
            return m.end()

        if cmd == "m":  # SGR (bold, reset, etc.)
            for p in (m.group(1) or "0").split(";"):
                p = p or "0"
                if p == "0":
                    console._ansi_fmt = QTextCharFormat()  # type: ignore[attr-defined]
                elif p == "1":
                    console._ansi_fmt.setFontWeight(QFont.Bold)
                elif p == "22":
                    console._ansi_fmt.setFontWeight(QFont.Normal)
            cursor.setCharFormat(console._ansi_fmt)
            return m.end()

        return None


class EmittingStream(QObject):
    textWritten: Signal = Signal(str)

    def write(self, data: str) -> None:
        data = str(data)
        if data:
            self.textWritten.emit(data)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return True


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
