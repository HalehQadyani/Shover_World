# test.py
from gui import ShoverWorldGUI

def main():
    gui = ShoverWorldGUI(map_path="maps/map1.txt")
    gui.run()

if __name__ == "__main__":
    main()
