"""
Меню выбора противника для Евразийских шашек.
Запуск: python play.py
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os


class MenuWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Евразийские шашки")
        self.geometry("420x500")
        self.resizable(False, False)
        self.configure(bg="#2b2b2b")

        self.selected_bot = tk.StringVar(value="random")

        # ── Заголовок ──────────────────────────────────────────────────────
        tk.Label(
            self, text="Евразийские шашки",
            font=("Arial", 20, "bold"),
            bg="#2b2b2b", fg="white"
        ).pack(pady=(30, 5))

        tk.Label(
            self, text="Выберите противника",
            font=("Arial", 13),
            bg="#2b2b2b", fg="#aaaaaa"
        ).pack(pady=(0, 25))

        # ── Карточки выбора бота ───────────────────────────────────────────
        bots = [
            (
                "random",
                "🎲  Случайный бот",
                "Ходит случайным образом.\nСамый лёгкий противник."
            ),
            (
                "minmax",
                "♟  Минимакс бот",
                "Просчитывает ходы на 2 шага вперёд.\nСредний уровень сложности."
            ),
            (
                "neural",
                "🧠  Нейросетевой бот (DQN)",
                "Обучен методом обучения\nс подкреплением (5000 партий)."
            ),
        ]

        for value, title, desc in bots:
            self._make_card(value, title, desc)

        # ── Кнопка старта ─────────────────────────────────────────────────
        tk.Button(
            self,
            text="▶  Начать игру",
            font=("Arial", 14, "bold"),
            bg="#4CAF50", fg="white",
            activebackground="#45a049",
            relief="flat", cursor="hand2",
            padx=20, pady=10,
            command=self._start
        ).pack(pady=25)

    def _make_card(self, value, title, desc):
        """Создаёт карточку-радиокнопку для выбора бота."""
        frame = tk.Frame(
            self, bg="#3c3c3c",
            relief="flat", bd=0,
            cursor="hand2"
        )
        frame.pack(fill="x", padx=30, pady=5)
        frame.bind("<Button-1>", lambda e: self.selected_bot.set(value))

        radio = tk.Radiobutton(
            frame,
            variable=self.selected_bot,
            value=value,
            bg="#3c3c3c",
            activebackground="#3c3c3c",
            cursor="hand2"
        )
        radio.grid(row=0, column=0, rowspan=2, padx=(10, 0), pady=8)

        tk.Label(
            frame, text=title,
            font=("Arial", 12, "bold"),
            bg="#3c3c3c", fg="white",
            anchor="w"
        ).grid(row=0, column=1, sticky="w", padx=8, pady=(8, 0))

        tk.Label(
            frame, text=desc,
            font=("Arial", 9),
            bg="#3c3c3c", fg="#aaaaaa",
            anchor="w", justify="left"
        ).grid(row=1, column=1, sticky="w", padx=8, pady=(0, 8))

        frame.bind("<Button-1>", lambda e: self.selected_bot.set(value))
        for child in frame.winfo_children():
            child.bind("<Button-1>", lambda e, v=value: self.selected_bot.set(v))

    def _start(self):
        bot = self.selected_bot.get()

        # Проверяем наличие модели для нейробота
        if bot == "neural":
            if not os.path.exists("dqn_model.pth") and \
               not os.path.exists("dqn_model_ep5000.pth"):
                messagebox.showerror(
                    "Модель не найдена",
                    "Файл dqn_model.pth не найден.\n\n"
                    "Сначала запустите обучение:\n"
                    "python dqn_checkers.py train"
                )
                return
            # Используем ep5000 если основная не найдена
            if not os.path.exists("dqn_model.pth"):
                os.rename("dqn_model_ep5000.pth", "dqn_model.pth")

        self.destroy()
        _launch_game(bot)


# ══════════════════════════════════════════════════════════════════════════════
#  Запуск игры с нужным ботом
# ══════════════════════════════════════════════════════════════════════════════

def _launch_game(bot_type):
    """Патчит board.py нужным ботом и запускает игру."""
    import importlib
    import board as board_module

    original_auto_turn = board_module.CheckersBoard.auto_turn

    if bot_type == "random":
        # Оставляем оригинальный случайный бот
        pass

    elif bot_type == "minmax":
        def minmax_auto_turn(self):
            """Минимакс бот: просчитывает позицию на 2 хода вперёд."""
            from dqn_checkers import MinMaxBot, EurasianCheckersEnv, CheckersPiece

            if not hasattr(self, '_minmax_bot'):
                self._minmax_bot = MinMaxBot(depth=2)

            # Строим объект среды из текущего состояния доски
            env = EurasianCheckersEnv.__new__(EurasianCheckersEnv)
            env.turn = EurasianCheckersEnv.OPPONENT  # ход белых (бот)
            env.move_count = 0
            env.done = False
            env.winner = None
            env.board = [[None]*8 for _ in range(8)]

            for r in range(8):
                for c in range(8):
                    p = self.cells[r][c]
                    if p is not None:
                        color = EurasianCheckersEnv.OPPONENT if p.color == self.autoColor \
                                else EurasianCheckersEnv.PLAYER
                        new_p = CheckersPiece(color, r, c)
                        new_p.is_queen = p.is_queen
                        env.board[r][c] = new_p

            moves = env._get_all_moves(env.turn)
            if not moves:
                return

            action_idx = self._minmax_bot.choose_action(env)
            action_idx = min(action_idx, len(moves) - 1)
            fr, fc, tr, tc = moves[action_idx]

            self.on_cell_click(fr, fc, self.autoColor)
            self.on_cell_click(tr, tc, self.autoColor)

        board_module.CheckersBoard.auto_turn = minmax_auto_turn

    elif bot_type == "neural":
        def neural_auto_turn(self):
            """Нейросетевой бот на основе обученной DQN-модели."""
            # Инициализируем бота один раз
            if not hasattr(self, '_neural_bot'):
                try:
                    from dqn_checkers import NeuralBot
                    self._neural_bot = NeuralBot('dqn_model.pth')
                    print("NeuralBot загружен.")
                except Exception as e:
                    print(f"Ошибка загрузки модели: {e}. Использую случайный бот.")
                    self._neural_bot = None

            if self._neural_bot is None:
                # Fallback на случайный бот
                original_auto_turn(self)
                return

            # Кодируем текущую доску в вектор состояния (33 числа)
            state = self._encode_state()
            # Получаем все допустимые ходы
            moves = self._get_neural_moves()
            if not moves:
                return

            n_actions = len(moves)
            action_idx = self._neural_bot.select_action(state, n_actions)
            action_idx = min(action_idx, n_actions - 1)

            fr, fc, tr, tc = moves[action_idx]
            self.on_cell_click(fr, fc, self.autoColor)
            self.on_cell_click(tr, tc, self.autoColor)

        def encode_state(self):
            """Кодирует доску в вектор из 33 чисел для нейросети."""
            import numpy as np
            state = []
            for r in range(8):
                for c in range(8):
                    if (r + c) % 2 == 1:  # тёмная клетка
                        p = self.cells[r][c]
                        if p is None:
                            state.append(0)
                        elif p.color == self.autoColor:
                            state.append(2 if p.is_queen else 1)
                        else:
                            state.append(-2 if p.is_queen else -1)
            state.append(0)  # ход бота (0 = противник с точки зрения env)
            return np.array(state, dtype=np.float32)

        def get_neural_moves(self):
            """Возвращает список ходов для бота в формате (fr,fc,tr,tc)."""
            import random
            captures = []
            normal = []

            for r in range(8):
                for c in range(8):
                    p = self.cells[r][c]
                    if p and p.color == self.autoColor:
                        caps = self.find_possibilities_to_attack(r, c)
                        if caps:
                            captures.extend([(r, c, nr, nc) for nr, nc in caps])
                        else:
                            # Обычные диагональные ходы
                            if p.is_queen:
                                for dr, dc in [(-1,1),(-1,-1),(1,1),(1,-1)]:
                                    nr, nc = r+dr, c+dc
                                    while 0<=nr<8 and 0<=nc<8:
                                        if self.cells[nr][nc] is None:
                                            normal.append((r, c, nr, nc))
                                        else:
                                            break
                                        nr+=dr; nc+=dc
                            else:
                                for dc in [-1, 1]:
                                    nr, nc = r+1, c+dc
                                    if 0<=nr<8 and 0<=nc<8 and self.cells[nr][nc] is None:
                                        normal.append((r, c, nr, nc))

            return captures if captures else normal

        board_module.CheckersBoard.auto_turn = neural_auto_turn
        board_module.CheckersBoard._encode_state = encode_state
        board_module.CheckersBoard._get_neural_moves = get_neural_moves

    # ── Запускаем игру ────────────────────────────────────────────────────
    from enter_window import EnterWindow
    from game import Game

    enter_window = EnterWindow()
    enter_window.mainloop()
    if not enter_window.response:
        sys.exit(1)

    game_window = Game()
    game_window.mainloop()


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    menu = MenuWindow()
    menu.mainloop()