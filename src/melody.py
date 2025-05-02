import sys

from MIDRC_MELODY.common.edit_config import edit_config
from MIDRC_MELODY.common.generate_eod_aaod_spiders import generate_eod_aaod_spiders
from MIDRC_MELODY.common.generate_qwk_spiders import generate_qwk_spiders

try:
    from melody_gui import launch_gui
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

def show_config(path):
    """Print out the YAML config file at `path`."""
    try:
        print(open(path, encoding='utf-8').read())
    except Exception as e:
        print(f"Error reading config: {e}")

def set_config(current_path):
    """Prompt for a new path and return it (or the old one)."""
    new = input(f"Enter new config path [{current_path}]: ").strip()
    return new or current_path

def quit_program(_=None):
    print("Goodbye!")
    sys.exit(0)

def main():
    # wrap your mutable state in a dict
    state = {"config_path": "config.yaml"}

    commands = {
        "1": ("Calculate QWK metrics",
              lambda: generate_qwk_spiders(cfg_path=state["config_path"])),
        "2": ("Calculate EOD/AAOD metrics",
              lambda: generate_eod_aaod_spiders(cfg_path=state["config_path"])),
    }
    if GUI_AVAILABLE:
        commands["3"] = ("Launch GUI", lambda: launch_gui())
    commands["s"] = ("Show current config file contents", lambda: show_config(state["config_path"]))
    commands["f"] = ("Change config file path", lambda: state.update(config_path=set_config(state["config_path"])))
    commands["e"] = ("Edit config file", lambda: edit_config(state["config_path"]))
    commands["q"] = ("Quit", quit_program)

    BOLD = '\033[1m'
    RESET = '\033[0m'
    while True:
        print("\n=== MIDRC‑MELODY Menu ===")
        print(f"Current config file path: {BOLD}{state['config_path']}{RESET}")
        for key, (desc, _) in commands.items():
            print(f"  {key}) {desc}")
        choice = input("Select an option: ").strip()

        action = commands.get(choice)
        if not action:
            print(f"  ❌  '{choice}' is not a valid choice.")
            continue

        # call the handler
        _, handler = action
        handler()

if __name__ == "__main__":
    main()
