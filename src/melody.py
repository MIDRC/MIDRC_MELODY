# python
from common.generate_eod_aaod_spiders import generate_eod_aaod_spiders
from common.generate_qwk_spiders import generate_qwk_spiders

def main():
    config_path = "config.yaml"
    bold_start = "\033[1m"
    bold_end = "\033[0m"

    while True:
        # Print application banner and current config file in bold
        print("\n*** MIDRC-MELODY ***")
        print(f"Current config file: {bold_start}{config_path}{bold_end}")
        print("Select an action:")
        print("1) Calculate EOD and AAOD metrics")
        print("2) Calculate QWK metrics")
        print("3) Print config file contents")
        print("4) Change config file")
        print("0) Exit")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            generate_eod_aaod_spiders(cfg_path=config_path)
        elif choice == "2":
            generate_qwk_spiders(cfg_path=config_path)
        elif choice == "3":
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    print("\nConfig file contents:")
                    print(f.read())
            except Exception as e:
                print(f"Failed to read config file: {e}")
            input("\nPress Enter to return to the menu...")
        elif choice == "4":
            new_path = input(f"Current config is {config_path}. Enter new config file path or press Enter to keep current: ").strip()
            if new_path:
                config_path = new_path
                print(f"Config file changed to: {config_path}")
            else:
                print("Config file remains unchanged.")
            input("\nPress Enter to return to the menu...")
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == '__main__':
    main()
