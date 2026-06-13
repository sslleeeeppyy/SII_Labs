import sys
from enter_window import EnterWindow
from game import Game


if __name__ == '__main__':
    enter_window = EnterWindow()
    enter_window.mainloop()
    if not enter_window.response:
        sys.exit(1)
    game_window = Game()
    game_window.mainloop()
