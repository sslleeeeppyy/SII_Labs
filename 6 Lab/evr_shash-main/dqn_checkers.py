"""
Лабораторная работа №6
DQN-агент для игры в Евразийские шашки
Автор: Вражкин Никита Александрович, ИСТбд-31
"""

import random
import numpy as np
from collections import deque
import json
import os

# ─── Попытка импорта torch ────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch не установлен. Будет использована эмуляция без обучения.")


# ══════════════════════════════════════════════════════════════════════════════
#  1. СРЕДА (Environment)
# ══════════════════════════════════════════════════════════════════════════════

class CheckersPiece:
    """Шашка: хранит цвет, координаты и флаг дамки."""
    def __init__(self, color, row, col):
        self.color = color      # 'white' или 'black'
        self.row = row
        self.col = col
        self.is_queen = False


class EurasianCheckersEnv:
    """
    Среда для Евразийских шашек (RL-окружение).

    Особенности:
    - 8×8 доска, чёрные клетки
    - Шашки ходят по диагонали вперёд
    - Бой происходит ортогонально (вертикально и горизонтально)
    - Дамка ходит на любое расстояние по диагонали/ортогонально
    """

    PLAYER = 'black'   # агент
    OPPONENT = 'white' # противник

    # Функция вознаграждения
    REWARD_CAPTURE_PIECE  = +10
    REWARD_CAPTURE_QUEEN  = +20
    REWARD_WIN            = +50
    REWARD_LOSE           = -50
    REWARD_STEP           = -0.05
    REWARD_ILLEGAL        = -1

    MAX_MOVES = 300       # максимальная длина партии

    def __init__(self):
        self.board = None
        self.reset()

    # ── Служебные ─────────────────────────────────────────────────────────────
    @staticmethod
    def _is_valid(r, c):
        return 0 <= r < 8 and 0 <= c < 8

    @staticmethod
    def _is_dark(r, c):
        """Тёмная клетка — (row+col) нечётное (как в реальных шашках)."""
        return (r + c) % 2 == 1

    # ── Инициализация ──────────────────────────────────────────────────────────
    def reset(self):
        """Сброс доски в начальное положение. Возвращает начальное состояние."""
        self.board = [[None]*8 for _ in range(8)]
        self.turn = self.PLAYER
        self.move_count = 0
        self.done = False
        self.winner = None

        # Расстановка шашек
        for r in range(8):
            for c in range(8):
                if self._is_dark(r, c):
                    if r < 2:
                        self.board[r][c] = CheckersPiece(self.OPPONENT, r, c)
                    elif r > 5:
                        self.board[r][c] = CheckersPiece(self.PLAYER, r, c)
        return self._get_state()

    # ── Кодирование состояния ─────────────────────────────────────────────────
    def _get_state(self):
        """
        Вектор состояния длиной 33:
        - 32 тёмных клетки: 0 = пусто, 1 = агент, 2 = дамка агента,
                            -1 = противник, -2 = дамка противника
        - 1 признак: чей ход (1 = агент, 0 = противник)
        """
        state = []
        for r in range(8):
            for c in range(8):
                if self._is_dark(r, c):
                    p = self.board[r][c]
                    if p is None:
                        state.append(0)
                    elif p.color == self.PLAYER:
                        state.append(2 if p.is_queen else 1)
                    else:
                        state.append(-2 if p.is_queen else -1)
        state.append(1 if self.turn == self.PLAYER else 0)
        return np.array(state, dtype=np.float32)

    # ── Генерация ходов ───────────────────────────────────────────────────────
    def _get_all_moves(self, color):
        """Все допустимые ходы для заданного цвета."""
        captures = []
        normal = []
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if p and p.color == color:
                    caps = self._get_captures(r, c)
                    if caps:
                        captures.extend([(r, c, nr, nc) for nr, nc in caps])
                    else:
                        moves = self._get_moves(r, c)
                        normal.extend([(r, c, nr, nc) for nr, nc in moves])
        # Если есть ходы-взятия — обязательны
        return captures if captures else normal

    def _get_moves(self, r, c):
        """Обычные ходы (без взятия) для шашки на (r,c)."""
        p = self.board[r][c]
        moves = []
        if p.is_queen:
            for dr, dc in [(-1,1),(-1,-1),(1,1),(1,-1)]:
                nr, nc = r+dr, c+dc
                while self._is_valid(nr, nc) and self.board[nr][nc] is None:
                    moves.append((nr, nc))
                    nr += dr; nc += dc
        else:
            direction = -1 if p.color == self.PLAYER else 1
            for dc in [-1, 1]:
                nr, nc = r + direction, c + dc
                if self._is_valid(nr, nc) and self.board[nr][nc] is None:
                    moves.append((nr, nc))
        return moves

    def _get_captures(self, r, c):
        """Ортогональные взятия для шашки на (r,c)."""
        p = self.board[r][c]
        captures = []
        if p.is_queen:
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                enemy_found = False
                nr, nc = r+dr, c+dc
                while self._is_valid(nr, nc):
                    cell = self.board[nr][nc]
                    if cell is None:
                        if enemy_found:
                            captures.append((nr, nc))
                    elif cell.color != p.color:
                        if enemy_found:
                            break
                        enemy_found = True
                    else:
                        break
                    nr += dr; nc += dc
        else:
            for dr, dc in [(0,2),(0,-2),(2,0),(-2,0)]:
                nr, nc = r+dr, c+dc
                mr, mc = r+dr//2, c+dc//2
                if (self._is_valid(nr, nc) and
                    self.board[nr][nc] is None and
                    self.board[mr][mc] is not None and
                    self.board[mr][mc].color != p.color):
                    captures.append((nr, nc))
        return captures

    # ── Применение хода ──────────────────────────────────────────────────────
    def step(self, action_idx):
        """
        Применяет ход по индексу из списка допустимых ходов.
        Возвращает (next_state, reward, done, info).
        """
        moves = self._get_all_moves(self.turn)
        if not moves:
            # Нет ходов — поражение
            self.done = True
            self.winner = self.OPPONENT if self.turn == self.PLAYER else self.PLAYER
            reward = self.REWARD_LOSE if self.turn == self.PLAYER else self.REWARD_WIN
            return self._get_state(), reward, True, {}

        if action_idx < 0 or action_idx >= len(moves):
            self.done = True
            return self._get_state(), self.REWARD_ILLEGAL, True, {'illegal': True}

        fr, fc, tr, tc = moves[action_idx]
        reward = self.REWARD_STEP
        piece = self.board[fr][fc]

        # Взятие
        captured_queen = False
        dr = tr - fr; dc = tc - fc
        if abs(dr) > 1 or abs(tc - fc) > 1:
            # Ищем взятую шашку
            if piece.is_queen:
                steps = max(abs(dr), abs(dc))
                sr = dr // steps; sc = dc // steps
                for i in range(1, steps):
                    cell = self.board[fr + i*sr][fc + i*sc]
                    if cell is not None:
                        captured_queen = cell.is_queen
                        self.board[fr + i*sr][fc + i*sc] = None
                        break
            else:
                mr = (fr + tr) // 2; mc = (fc + tc) // 2
                if self.board[mr][mc]:
                    captured_queen = self.board[mr][mc].is_queen
                    self.board[mr][mc] = None
            reward += self.REWARD_CAPTURE_QUEEN if captured_queen else self.REWARD_CAPTURE_PIECE

        # Перемещение
        self.board[fr][fc] = None
        piece.row = tr; piece.col = tc
        self.board[tr][tc] = piece

        # Превращение в дамку
        if not piece.is_queen:
            if piece.color == self.PLAYER and tr == 0:
                piece.is_queen = True
            elif piece.color == self.OPPONENT and tr == 7:
                piece.is_queen = True

        # Проверка завершения
        self.move_count += 1
        if self._check_win(self.PLAYER):
            self.done = True; self.winner = self.PLAYER
            reward += self.REWARD_WIN
        elif self._check_win(self.OPPONENT):
            self.done = True; self.winner = self.OPPONENT
            reward += self.REWARD_LOSE
        elif self.move_count >= self.MAX_MOVES:
            self.done = True; self.winner = 'draw'

        # Смена хода
        if not self.done:
            self.turn = self.OPPONENT if self.turn == self.PLAYER else self.PLAYER

        return self._get_state(), reward, self.done, {}

    def _check_win(self, color):
        """Проверяет, есть ли шашки у противника."""
        opponent = self.OPPONENT if color == self.PLAYER else self.PLAYER
        has_pieces = any(
            self.board[r][c] and self.board[r][c].color == opponent
            for r in range(8) for c in range(8)
        )
        if not has_pieces:
            return True
        # Нет ходов у противника
        if not self._get_all_moves(opponent):
            return True
        return False

    def get_action_count(self):
        return len(self._get_all_moves(self.turn))

    def get_state_size(self):
        return 33


# ══════════════════════════════════════════════════════════════════════════════
#  2. НЕЙРОННАЯ СЕТЬ (Q-network)
# ══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        """
        Полносвязная сеть:
        вход(33) → 128 → 128 → 64 → выход(200)
        Активация ReLU, выходной слой — линейный.
        """
        def __init__(self, state_size=33, action_size=200):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_size)
            )

        def forward(self, x):
            return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
#  3. DQN-АГЕНТ
# ══════════════════════════════════════════════════════════════════════════════

class DQNAgent:
    """
    DQN-агент с:
    - буфером воспроизведения (replay buffer) размером 20 000
    - целевой сетью (target network), обновляемой каждые 1000 шагов
    - ε-жадной стратегией (epsilon от 1.0 до 0.01 за 50 000 шагов)
    """
    STATE_SIZE   = 33
    ACTION_SIZE  = 200
    BUFFER_SIZE  = 20_000
    BATCH_SIZE   = 64
    GAMMA        = 0.95    # коэффициент дисконтирования (снижен для стабильности)
    LR           = 0.0001  # learning rate Adam (снижен для предотвращения взрыва)
    EPS_START    = 1.0
    EPS_END      = 0.01
    EPS_STEPS    = 50_000  # число шагов для линейного уменьшения ε
    TARGET_UPDATE = 1_000  # каждые N шагов обновляем целевую сеть

    def __init__(self):
        self.epsilon = self.EPS_START
        self.steps_done = 0
        self.memory = deque(maxlen=self.BUFFER_SIZE)

        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.policy_net = QNetwork(self.STATE_SIZE, self.ACTION_SIZE).to(self.device)
            self.target_net = QNetwork(self.STATE_SIZE, self.ACTION_SIZE).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR)
            self.loss_fn = nn.MSELoss()

    def _update_epsilon(self):
        """Линейное уменьшение ε от EPS_START до EPS_END за EPS_STEPS шагов."""
        if self.steps_done < self.EPS_STEPS:
            self.epsilon = self.EPS_START - (self.EPS_START - self.EPS_END) * \
                           (self.steps_done / self.EPS_STEPS)
        else:
            self.epsilon = self.EPS_END

    def select_action(self, state, n_actions):
        """ε-жадная стратегия: случайный ход или лучший по Q."""
        self.steps_done += 1
        self._update_epsilon()
        if random.random() < self.epsilon or not TORCH_AVAILABLE:
            return random.randint(0, max(0, n_actions - 1))
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.policy_net(s)[0]
            # Маскируем недопустимые индексы
            mask = torch.full((self.ACTION_SIZE,), float('-inf'))
            mask[:n_actions] = q[:n_actions]
            return int(mask.argmax().item())

    def remember(self, state, action, reward, next_state, done):
        # Обрезаем награду в диапазон [-1, 1] для стабильности
        reward = max(-1.0, min(1.0, reward / 50.0))
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Обучение на случайном батче из буфера воспроизведения."""
        if not TORCH_AVAILABLE or len(self.memory) < self.BATCH_SIZE:
            return 0.0

        batch = random.sample(self.memory, self.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        s  = torch.FloatTensor(np.array(states)).to(self.device)
        ns = torch.FloatTensor(np.array(next_states)).to(self.device)
        a  = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device)
        d  = torch.FloatTensor(dones).to(self.device)

        # Текущие Q-значения
        current_q = self.policy_net(s).gather(1, a).squeeze(1)

        # Целевые Q-значения (по целевой сети)
        with torch.no_grad():
            max_next_q = self.target_net(ns).max(1)[0]
            target_q = r + self.GAMMA * max_next_q * (1 - d)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — обрезает слишком большие градиенты
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target_network(self):
        """Копирует веса policy_net в target_net."""
        if TORCH_AVAILABLE:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path='dqn_model.pth'):
        if TORCH_AVAILABLE:
            torch.save(self.policy_net.state_dict(), path)
            print(f"Модель сохранена: {path}")

    def load(self, path='dqn_model.pth'):
        if TORCH_AVAILABLE and os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Модель загружена: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  4. БАЗОВЫЕ БОТЫ (для сравнения)
# ══════════════════════════════════════════════════════════════════════════════

class RandomBot:
    """Бот, выбирающий случайный допустимый ход."""
    def select_action(self, state, n_actions):
        return random.randint(0, max(0, n_actions - 1))


class MinMaxBot:
    """
    Бот на основе алгоритма Минимакс с заданной глубиной поиска.

    Минимакс — алгоритм перебора дерева ходов:
    - Максимизирующий игрок (агент) выбирает ход с наибольшей оценкой.
    - Минимизирующий игрок (противник) выбирает ход с наименьшей оценкой.
    - Глубина depth определяет сколько ходов вперёд просматривается.

    Оценочная функция:
    +10 за каждую шашку агента, +20 за дамку агента,
    -10 за каждую шашку противника, -20 за дамку противника.
    """

    def __init__(self, depth=2):
        self.depth = depth  # глубина поиска (2 = смотрим 2 хода вперёд)

    def select_action(self, state, n_actions):
        return 0  # заглушка — MinMaxBot работает напрямую через env

    def choose_action(self, env):
        """Выбирает лучший ход через Минимакс. Принимает объект среды."""
        moves = env._get_all_moves(env.turn)
        if not moves:
            return 0

        best_score = float('-inf')
        best_idx = 0

        for i, move in enumerate(moves):
            # Делаем копию среды и применяем ход
            env_copy = self._copy_env(env)
            env_copy.step(i)
            # Оцениваем позицию через минимакс
            score = self._minimax(env_copy, self.depth - 1, False)
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def _minimax(self, env, depth, is_maximizing):
        """Рекурсивный минимакс."""
        if depth == 0 or env.done:
            return self._evaluate(env)

        moves = env._get_all_moves(env.turn)
        if not moves:
            return self._evaluate(env)

        if is_maximizing:
            best = float('-inf')
            for i in range(len(moves)):
                env_copy = self._copy_env(env)
                env_copy.step(i)
                best = max(best, self._minimax(env_copy, depth - 1, False))
            return best
        else:
            best = float('inf')
            for i in range(len(moves)):
                env_copy = self._copy_env(env)
                env_copy.step(i)
                best = min(best, self._minimax(env_copy, depth - 1, True))
            return best

    def _evaluate(self, env):
        """Оценочная функция позиции: считает материальное преимущество агента."""
        score = 0
        for r in range(8):
            for c in range(8):
                p = env.board[r][c]
                if p is None:
                    continue
                value = 20 if p.is_queen else 10
                if p.color == EurasianCheckersEnv.PLAYER:
                    score += value
                else:
                    score -= value
        return score

    def _copy_env(self, env):
        """Создаёт глубокую копию среды для симуляции."""
        import copy
        new_env = EurasianCheckersEnv.__new__(EurasianCheckersEnv)
        new_env.turn = env.turn
        new_env.move_count = env.move_count
        new_env.done = env.done
        new_env.winner = env.winner
        # Копируем доску
        new_env.board = [[None]*8 for _ in range(8)]
        for r in range(8):
            for c in range(8):
                p = env.board[r][c]
                if p is not None:
                    new_p = CheckersPiece(p.color, p.row, p.col)
                    new_p.is_queen = p.is_queen
                    new_env.board[r][c] = new_p
        return new_env


# ══════════════════════════════════════════════════════════════════════════════
#  5. ЦИКЛ ОБУЧЕНИЯ
# ══════════════════════════════════════════════════════════════════════════════

def train(n_episodes=20_000, save_every=2_000, verbose_every=500):
    """
    Основной цикл обучения DQN-агента.

    Улучшения:
    1. Сохранение лучшей модели по проценту побед (скользящее окно 500 эп.)
    2. Curriculum learning: RandomBot → MinMaxBot
    3. Более плавное уменьшение ε (за 100 000 шагов, не 50 000)
    4. Минимальный ε = 0.05 (а не 0.01) — сохраняем немного исследования
       чтобы избежать переобучения на узком наборе ситуаций
    """
    env = EurasianCheckersEnv()
    agent = DQNAgent()

    # Делаем ε-расписание более плавным
    agent.EPS_STEPS = 100_000
    agent.EPS_END = 0.05

    random_opponent = RandomBot()
    minmax_opponent = MinMaxBot(depth=2)

    stats = {'wins': 0, 'losses': 0, 'draws': 0,
             'captures': [], 'lengths': [], 'losses_list': [],
             'wins_window': deque(maxlen=verbose_every)}

    best_winrate = 0.0
    best_ep = 0

    print(f"Начало обучения. Устройство: "
          f"{'CUDA' if TORCH_AVAILABLE and torch.cuda.is_available() else 'CPU/no-torch'}")
    print(f"Эпизодов: {n_episodes}")
    print(f"Эп 1–3000:     противник RandomBot (изучение базовых ходов)")
    print(f"Эп 3001–{n_episodes}: противник MinMaxBot (изучение стратегии)")
    print(f"ε: 1.0 → 0.05 за 100000 шагов")
    print(f"Буфер: {DQNAgent.BUFFER_SIZE}, batch: {DQNAgent.BATCH_SIZE}")
    print(f"Лучшая модель сохраняется в dqn_model_best.pth\n")

    for ep in range(1, n_episodes + 1):
        # Curriculum: 3000 эпизодов на RandomBot, остальные на MinMax
        opponent = random_opponent if ep <= 3000 else minmax_opponent
        if ep == 3001:
            print("\n>>> Переключение на MinMaxBot <<<\n")

        state = env.reset()
        ep_captures = 0
        ep_loss = 0.0
        loss_count = 0

        while not env.done:
            n_act = env.get_action_count()

            if env.turn == EurasianCheckersEnv.PLAYER:
                # Ход агента
                action = agent.select_action(state, n_act)
                next_state, reward, done, info = env.step(action)
                if reward > 5:
                    ep_captures += 1
                agent.remember(state, action, reward, next_state, done)
                l = agent.replay()
                if l:
                    ep_loss += l; loss_count += 1
                state = next_state

                if agent.steps_done % DQNAgent.TARGET_UPDATE == 0:
                    agent.update_target_network()
            else:
                # Ход противника
                if isinstance(opponent, MinMaxBot):
                    action = opponent.choose_action(env)
                else:
                    action = opponent.select_action(state, n_act)
                next_state, _, done, _ = env.step(action)
                state = next_state

        # Статистика эпизода
        if env.winner == EurasianCheckersEnv.PLAYER:
            stats['wins'] += 1
            stats['wins_window'].append(1)
        elif env.winner == EurasianCheckersEnv.OPPONENT:
            stats['losses'] += 1
            stats['wins_window'].append(0)
        else:
            stats['draws'] += 1
            stats['wins_window'].append(0)
        stats['captures'].append(ep_captures)
        stats['lengths'].append(env.move_count)
        stats['losses_list'].append(ep_loss / max(1, loss_count))

        # Сохранение лучшей модели по скользящему окну
        if ep >= verbose_every and len(stats['wins_window']) == verbose_every:
            current_winrate = sum(stats['wins_window']) / verbose_every
            if current_winrate > best_winrate:
                best_winrate = current_winrate
                best_ep = ep
                agent.save('dqn_model_best.pth')

        if ep % verbose_every == 0:
            w = stats['wins']; l = stats['losses']; d = stats['draws']
            total_ep = w + l + d
            recent_winrate = sum(stats['wins_window']) / len(stats['wins_window']) * 100
            avg_cap = np.mean(stats['captures'][-verbose_every:])
            avg_len = np.mean(stats['lengths'][-verbose_every:])
            avg_loss = np.mean(stats['losses_list'][-verbose_every:])
            opp_name = "Random" if ep <= 3000 else "MinMax"
            print(f"Эп {ep:6d} [{opp_name}] | "
                  f"Побед(500): {recent_winrate:5.1f}% | "
                  f"Взятий: {avg_cap:.1f} | "
                  f"Ходов: {avg_len:.0f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ε={agent.epsilon:.3f} | "
                  f"★ Best: {best_winrate*100:.1f}% (эп {best_ep})")

        if ep % save_every == 0:
            agent.save(f'dqn_model_ep{ep}.pth')

    agent.save('dqn_model.pth')
    with open('train_stats.json', 'w') as f:
        json.dump({k: (v if isinstance(v, (int,)) else [float(x) for x in v])
                   for k, v in stats.items() if k != 'wins_window'}, f)
    print(f"\nОбучение завершено.")
    print(f"Лучшая модель: dqn_model_best.pth (win rate {best_winrate*100:.1f}% на эп {best_ep})")
    print(f"Финальная модель: dqn_model.pth")
    return agent, stats


# ══════════════════════════════════════════════════════════════════════════════
#  6. НЕЙРОБОТ (NeuralBot) — использует обученную модель
# ══════════════════════════════════════════════════════════════════════════════

class NeuralBot:
    """
    Бот на основе обученной DQN-модели.
    Используется в реальной игре вместо RandomBot.
    """
    def __init__(self, model_path='dqn_model.pth'):
        self.agent = DQNAgent()
        self.agent.epsilon = 0.0   # полностью жадный
        self.agent.load(model_path)

    def select_action(self, state, n_actions):
        return self.agent.select_action(state, n_actions)


# ══════════════════════════════════════════════════════════════════════════════
#  7. СРАВНЕНИЕ АГЕНТОВ
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(agent, opponent, n_games=200):
    """Оценивает агента против противника за n_games партий."""
    env = EurasianCheckersEnv()
    wins = losses = draws = 0
    total_captures = 0
    total_length = 0

    for _ in range(n_games):
        state = env.reset()
        captures = 0
        while not env.done:
            n_act = env.get_action_count()
            if env.turn == EurasianCheckersEnv.PLAYER:
                # MinMaxBot использует choose_action(env), остальные select_action
                if isinstance(agent, MinMaxBot):
                    action = agent.choose_action(env)
                else:
                    action = agent.select_action(state, n_act)
            else:
                if isinstance(opponent, MinMaxBot):
                    action = opponent.choose_action(env)
                else:
                    action = opponent.select_action(state, n_act)
            next_state, reward, done, _ = env.step(action)
            if reward > 5:
                captures += 1
            state = next_state
        if env.winner == EurasianCheckersEnv.PLAYER:
            wins += 1
        elif env.winner == EurasianCheckersEnv.OPPONENT:
            losses += 1
        else:
            draws += 1
        total_captures += captures
        total_length += env.move_count

    return {
        'wins_pct':    wins   / n_games * 100,
        'losses_pct':  losses / n_games * 100,
        'draws_pct':   draws  / n_games * 100,
        'avg_captures': total_captures / n_games,
        'avg_length':   total_length   / n_games,
    }


def run_comparison():
    """Сравнивает NeuralBot, RandomBot и MinMaxBot."""
    minmax_bot = MinMaxBot(depth=2)

    bots = {
        'RandomBot': RandomBot(),
        'MinMaxBot': minmax_bot,
    }
    # Приоритет: best > основная > ep_N
    model_path = None
    if os.path.exists('dqn_model_best.pth'):
        model_path = 'dqn_model_best.pth'
    elif os.path.exists('dqn_model.pth'):
        model_path = 'dqn_model.pth'

    if model_path:
        print(f"Используется модель: {model_path}")
        bots['NeuralBot'] = NeuralBot(model_path)

    print("\n=== Сравнение агентов (200 партий каждого против RandomBot) ===")
    print(f"{'Агент':<12} {'Победы':>8} {'Поражения':>11} {'Ничьи':>8} "
          f"{'Взятий/парт':>12} {'Ходов/парт':>11}")
    print("-" * 65)

    for name, bot in bots.items():
        r = evaluate(bot, RandomBot(), n_games=200)
        print(f"{name:<12} {r['wins_pct']:>7.1f}% {r['losses_pct']:>10.1f}% "
              f"{r['draws_pct']:>7.1f}% {r['avg_captures']:>12.1f} "
              f"{r['avg_length']:>11.1f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Точка входа
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'

    if mode == 'train':
        train(n_episodes=20_000, save_every=2_000, verbose_every=500)
        run_comparison()
    elif mode == 'eval':
        run_comparison()
    else:
        print("Использование: python dqn_checkers.py [train|eval]")