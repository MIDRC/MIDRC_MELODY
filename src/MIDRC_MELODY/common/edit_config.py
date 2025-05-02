import curses
from curses.textpad import Textbox, rectangle


class PadWrapper:
    def __init__(self, pad, win_h, win_w, top, left):
        self._pad = pad
        self.offset = 0
        self.win_h = win_h
        self.win_w = win_w
        # screen coords where the pad is drawn
        self.top, self.left = top, left

    def refresh(self, *args):
        # ignore whatever args Textbox might pass
        self._pad.refresh(
            self.offset, 0,
            self.top, self.left,
            self.top + self.win_h - 1,
            self.left + self.win_w - 1
        )

    def __getattr__(self, name):
        # delegate everything else to the real pad
        return getattr(self._pad, name)


def edit_config(path):
    # Load the file
    with open(path, 'r', encoding='utf-8') as f:
        original = f.read().split('\n')

    def _inner(stdscr):
        # Add this to the screen in case curses doesn't finish cleaning up on ctrl-c
        print("Press Enter to Continue...")

        # — basic curses setup —
        curses.cbreak();
        curses.noecho();
        stdscr.keypad(True)
        curses.curs_set(1)
        h, w = stdscr.getmaxyx()
        win_h, win_w = h - 4, w - 4

        # — draw border & instructions —
        stdscr.clear()
        rectangle(stdscr, 1, 1, h - 2, w - 2)
        stdscr.addstr(h - 1, 2,
                      "Ctrl‑G=save  Ctrl‑C=cancel  ↑/↓=scroll  PgUp/PgDn=jump"
                      )
        stdscr.refresh()

        # — create real pad & wrap it —
        real_pad = curses.newpad(max(len(original) + 1, win_h), win_w)
        pad = PadWrapper(real_pad, win_h, win_w, top=2, left=2)
        pad.keypad(True)
        pad.scrollok(True)
        pad.idlok(True)

        # — fill with existing lines —
        for i, line in enumerate(original):
            try:
                pad.addstr(i, 0, line)
            except curses.error:
                pass

        # — initial draw via our wrapper —
        pad.move(0, 0)
        pad.refresh()

        # — validator to scroll when cursor hits edges —
        def validator(ch):
            y, x = pad.getyx()
            if ch in (curses.KEY_DOWN,):
                if y - pad.offset >= win_h - 1 and pad.offset < real_pad.getmaxyx()[0] - win_h:
                    pad.offset += 1
            elif ch in (curses.KEY_UP,):
                if y - pad.offset <= 0 and pad.offset > 0:
                    pad.offset -= 1
            elif ch == curses.KEY_NPAGE:  # Page Down
                pad.offset = min(pad.offset + win_h,
                                 real_pad.getmaxyx()[0] - win_h)
                if y < pad.offset:
                    pad.move(pad.offset, x)
            elif ch == curses.KEY_PPAGE:  # Page Up
                pad.offset = max(pad.offset - win_h, 0)
                if y > pad.offset + win_h - 1:
                    pad.move(pad.offset + win_h - 1, x)

            pad.refresh()
            return ch

        # — launch the editor —
        tb = Textbox(pad)
        try:
            tb.edit(validator)
        except KeyboardInterrupt:
            return  # cancelled
        finally:
            curses.flushinp()

        # — gather & write back —
        lines = []
        for i in range(real_pad.getmaxyx()[0]):
            raw = pad.instr(i, 0, win_w).decode('utf-8', 'ignore')
            lines.append(raw.rstrip('\x00'))
        with open(path, 'w', encoding='utf-8') as out:
            out.write("\n".join(lines))

    curses.wrapper(_inner)