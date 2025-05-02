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

import curses
from curses.textpad import rectangle, Textbox


class PadWrapper:
    def __init__(self, pad, win_h, win_w, top, left):
        self.real_pad = pad
        self.offset = 0
        self.win_h = win_h
        self.win_w = win_w
        self.top = top
        self.left = left

    def refresh(self, *args):
        self.real_pad.refresh(
            self.offset, 0,
            self.top, self.left,
            self.top + self.win_h - 1,
            self.left + self.win_w - 1
        )

    def __getattr__(self, name):
        return getattr(self.real_pad, name)


def _load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().split('\n')


def _save_file(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _handle_scroll(key, pad):
    real_h, _ = pad.real_pad.getmaxyx()
    win_h = pad.win_h
    y, x = pad.getyx()
    if key == curses.KEY_DOWN:
        if y - pad.offset >= win_h - 1 and pad.offset < real_h - win_h:
            pad.offset += 1
    elif key == curses.KEY_UP:
        if y - pad.offset <= 0 and pad.offset > 0:
            pad.offset -= 1
    elif key == curses.KEY_NPAGE:
        pad.offset = min(pad.offset + win_h, real_h - win_h)
        if y < pad.offset:
            pad.move(pad.offset, x)
    elif key == curses.KEY_PPAGE:
        pad.offset = max(pad.offset - win_h, 0)
        if y >= pad.offset + win_h:
            pad.move(pad.offset + win_h - 1, x)

    pad.refresh()
    return key


class ConfigEditor:
    def __init__(self, stdscr, original, path):
        self.stdscr = stdscr
        self.original = original
        self.path = path
        self._init_curses()
        self._draw_border()
        self.pad = self._make_pad()
        self._fill_pad()

    def _init_curses(self):
        curses.cbreak()
        curses.noecho()
        self.stdscr.keypad(True)
        curses.curs_set(1)

    def _draw_border(self):
        h, w = self.stdscr.getmaxyx()
        rectangle(self.stdscr, 1, 1, h - 2, w - 2)
        self.stdscr.addstr(
            h - 1, 2,
            "Ctrl-G=save  Ctrl-C=cancel  ↑/↓=scroll  PgUp/PgDn=jump"
        )
        self.stdscr.refresh()

    def _make_pad(self):
        h, w = self.stdscr.getmaxyx()
        win_h, win_w = h - 4, w - 4
        real_h = max(len(self.original) + 1, win_h)
        _real_pad = curses.newpad(real_h, win_w)
        pad = PadWrapper(_real_pad, win_h, win_w, top=2, left=2)
        pad.keypad(True)
        pad.scrollok(True)
        pad.idlok(True)
        return pad

    def _fill_pad(self):
        for idx, line in enumerate(self.original):
            try:
                self.pad.addstr(idx, 0, line)
            except curses.error:
                pass
        self.pad.move(0, 0)
        self.pad.refresh()

    def _collect_lines(self):
        real_h, _ = self.pad.real_pad.getmaxyx()
        lines = []
        for i in range(real_h):
            raw = self.pad.instr(i, 0, self.pad.win_w).decode('utf-8', 'ignore')
            lines.append(raw.rstrip('\x00'))
        return lines

    def _validator(self, ch):
        _handle_scroll(ch, self.pad)
        try:
            self.pad.refresh()
        except curses.error:
            pass
        return ch

    def run(self):
        tb = Textbox(self.pad)
        try:
            tb.edit(self._validator)
        except KeyboardInterrupt:
            return
        finally:
            curses.flushinp()
        lines = self._collect_lines()
        _save_file(self.path, lines)


def _run_editor(stdscr, original, path):
    editor = ConfigEditor(stdscr, original, path)
    editor.run()


def edit_config(path):
    # Add this to the screen in case curses doesn't finish cleaning up on ctrl-c
    print("Press Any Key to Continue...")
    original = _load_file(path)
    curses.wrapper(lambda stdscr: _run_editor(stdscr, original, path))
