"""ANSI escape sequence processor and worker/thread helpers for GUI."""

import re
from typing import Final, Optional

from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtGui import QFont, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit

__all__ = ["ANSIProcessor", "EmittingStream", "Worker"]

_CSI_RE: Final[re.Pattern] = re.compile(r"\x1b\[(\d*)([A-Za-z])")


def _next_csi_is_cursor_up(buf: str, pos: int) -> bool:
    """Return True if next CSI sequence at pos is 'cursor up' (A)."""
    if pos >= len(buf) or buf[pos] != "\x1b":
        return False
    match = _CSI_RE.match(buf, pos)
    return bool(match and match.group(2) == "A")


class ANSIProcessor:
    """Process ANSI control sequences and render them in a QPlainTextEdit."""

    @staticmethod
    def process(console: QPlainTextEdit, chunk: str) -> None:
        """
        Parse chunk for ANSI escapes, updating the console widget.

        Args:
            console: Target QPlainTextEdit.
            chunk: Text with possible ANSI sequences.
        """
        ANSIProcessor._ensure_console_flags(console)
        cursor = console.textCursor()
        i, n = 0, len(chunk)
        text_buf: list[str] = []

        while i < n:
            if console._nl_pending:  # type: ignore[attr-defined]
                ANSIProcessor._commit_pending_newline(console, cursor, chunk, i)

            ch = chunk[i]
            if ch == "\r":  # Handle carriage return
                if text_buf:
                    cursor.insertText("".join(text_buf))
                    text_buf.clear()
                ANSIProcessor._handle_carriage_return(cursor)
                i += 1
                continue

            if ch == "\n":  # Handle line feed
                if text_buf:
                    cursor.insertText("".join(text_buf))
                    text_buf.clear()
                ANSIProcessor._handle_line_feed(cursor)
                i += 1
                continue

            if ch == "\x1b":  # Handle CSI (Control Sequence Introducer) sequence
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
        """Initialize internal flags for ANSI processing on the widget."""
        if not hasattr(console, "_nl_pending"):
            console._nl_pending = False  # type: ignore[attr-defined]
        if not hasattr(console, "_ansi_fmt"):
            console._ansi_fmt = QTextCharFormat()  # type: ignore[attr-defined]

    @staticmethod
    def _commit_pending_newline(
        console: QPlainTextEdit, cursor: QTextCursor, buf: str, pos: int
    ) -> None:
        """Insert a newline unless the next sequence is cursor-up."""
        if not _next_csi_is_cursor_up(buf, pos):
            cursor.insertBlock()
        console._nl_pending = False  # type: ignore[attr-defined]

    @staticmethod
    def _handle_carriage_return(cursor: QTextCursor) -> None:
        """Handle '\\r': move to start of line and clear it."""
        cursor.movePosition(QTextCursor.StartOfLine)
        cursor.select(QTextCursor.LineUnderCursor)
        cursor.removeSelectedText()

    @staticmethod
    def _handle_line_feed(cursor: QTextCursor) -> None:
        """Handle '\\n': move down or insert a new block then start of line."""
        if not cursor.movePosition(QTextCursor.Down):
            cursor.insertBlock()
        cursor.movePosition(QTextCursor.StartOfLine)

    @staticmethod
    def _handle_csi(
        console: QPlainTextEdit,
        cursor: QTextCursor,
        buf: str,
        pos: int,
    ) -> Optional[int]:
        """
        Process CSI (Control Sequence Introducer) sequences.

        Returns:
            Position after CSI if handled, otherwise None.
        """
        match = _CSI_RE.match(buf, pos)
        if not match:
            return None

        count = int(match.group(1) or 1)
        cmd = match.group(2)

        if cmd == "A":  # Cursor Up
            for _ in range(count):
                cursor.movePosition(QTextCursor.Up)
            cursor.movePosition(QTextCursor.StartOfLine)
            return match.end()

        if cmd == "K":  # Erase Line
            cursor.select(QTextCursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.movePosition(QTextCursor.StartOfLine)
            return match.end()

        if cmd == "m":  # Set Graphics Rendition (text style: e.g. bold, reset, etc.)
            for part in (match.group(1) or "0").split(";"):
                style = part or "0"
                if style == "0":
                    console._ansi_fmt = QTextCharFormat()  # type: ignore[attr-defined]
                elif style == "1":
                    console._ansi_fmt.setFontWeight(QFont.Bold)
                elif style == "22":
                    console._ansi_fmt.setFontWeight(QFont.Normal)
            cursor.setCharFormat(console._ansi_fmt)
            return match.end()

        return None


class EmittingStream(QObject):
    """A text stream that emits data via Qt signals."""

    textWritten: Signal = Signal(str)

    def write(self, data: str) -> None:
        """Emit chunk of text written to this stream."""
        text = str(data)
        if text:
            self.textWritten.emit(text)

    def flush(self) -> None:
        """No-op flush to satisfy stream interface."""
        pass

    def isatty(self) -> bool:
        """Indicate this stream behaves like a terminal."""
        return True


class WorkerSignals(QObject):
    """Signals for Worker: finished, error, and result."""
    finished: Signal = Signal()
    error: Signal = Signal(str)
    result: Signal = Signal(object)


class Worker(QRunnable):
    """
    QRunnable wrapper to execute a function in a separate thread.

    Args:
        fn: Callable to run.
        *args: Positional args for fn.
        **kwargs: Keyword args for fn.
    """

    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        """Execute the function and emit result, error, and finished signals."""
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
