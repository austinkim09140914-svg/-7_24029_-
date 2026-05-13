# app.py
# Gradio 웹 배포용 Bomber Escape Q-learning 프로그램
# 실행: python app.py

import random
from collections import defaultdict

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────
# 설정값
# ─────────────────────────────────────
GRID_SIZE = 10
CELL_SIZE = 56
BOARD_WIDTH = GRID_SIZE * CELL_SIZE
INFO_HEIGHT = 230
WIDTH = BOARD_WIDTH
HEIGHT = BOARD_WIDTH + INFO_HEIGHT

MAX_STEPS = 100
BOMB_PERIOD = 6
BLAST_RANGE = 2
PRETRAIN_EPISODES = 1000

ACTIONS = {
    0: (0, -1),   # up
    1: (0, 1),    # down
    2: (-1, 0),   # left
    3: (1, 0),    # right
    4: (0, 0),    # stay
}
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}

WHITE = (245, 245, 245)
BLACK = (30, 30, 30)
GRAY = (180, 180, 180)
DARK_GRAY = (80, 80, 80)
PANEL_BG = (238, 240, 245)
BLUE = (70, 130, 255)
GREEN = (80, 180, 120)
RED = (220, 80, 80)
YELLOW = (240, 190, 60)
ORANGE = (245, 130, 40)


def load_font(size=14, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf" if bold else "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "malgun.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

FONT_B = load_font(18, True)
FONT_M = load_font(15)
FONT_S = load_font(13)
FONT_Q = load_font(11)


# ─────────────────────────────────────
# Q-learning Agent
# ─────────────────────────────────────
class QLearningAgent:
    def __init__(self):
        self.action_size = 5
        self.alpha = 0.2
        self.gamma = 0.98
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.999
        self.q_table = defaultdict(lambda: [0.0] * self.action_size)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return self.best_action(state)

    def best_action(self, state):
        q_values = self.q_table[state]
        max_q = max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        next_max_q = 0 if done else max(self.q_table[next_state])
        target = reward + self.gamma * next_max_q
        self.q_table[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def get_q_values(self, state):
        return self.q_table[state]


# ─────────────────────────────────────
# 환경
# ─────────────────────────────────────
class BomberEscapeEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.walls = self._generate_fixed_walls()
        self.bomb_positions = [(2, 3), (7, 2), (4, 6), (8, 7)]
        self.reset()

    def _generate_fixed_walls(self):
        walls = set()
        patterns = [
            [(1, 1), (2, 1), (3, 1)],
            [(6, 1), (6, 2)],
            [(8, 2), (8, 3)],
            [(2, 4), (3, 4), (4, 4)],
            [(6, 5), (7, 5)],
            [(1, 6), (1, 7)],
            [(3, 7), (4, 7), (5, 7)],
            [(8, 8), (7, 8)],
        ]
        for group in patterns:
            for pos in group:
                walls.add(pos)
        return walls

    def _random_empty_pos(self, exclude=None):
        exclude = exclude or set()
        while True:
            pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if pos not in self.walls and pos not in exclude and pos not in self.bomb_positions:
                return pos

    def reset(self):
        placed = set(self.bomb_positions)
        self.runner = self._random_empty_pos(placed)
        placed.add(self.runner)
        self.exit = self._random_empty_pos(placed)
        self.bomb_phases = {pos: i % BOMB_PERIOD for i, pos in enumerate(self.bomb_positions)}
        self.steps = 0
        self._prev_dist_exit = self._manhattan(self.runner, self.exit)
        self._prev_danger_dist = self._nearest_bomb_dist(self.runner)
        return self.get_state()

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _nearest_bomb(self, pos):
        return min(self.bomb_positions, key=lambda p: self._manhattan(pos, p))

    def _nearest_bomb_dist(self, pos):
        return min(self._manhattan(pos, p) for p in self.bomb_positions)

    def _blocked_line(self, a, b):
        ax, ay = a
        bx, by = b
        if ax == bx:
            step = 1 if by > ay else -1
            for y in range(ay + step, by, step):
                if (ax, y) in self.walls:
                    return True
        elif ay == by:
            step = 1 if bx > ax else -1
            for x in range(ax + step, bx, step):
                if (x, ay) in self.walls:
                    return True
        return False

    def _is_exploding(self, bomb_pos):
        return self.bomb_phases[bomb_pos] == BOMB_PERIOD - 1

    def _in_blast(self, pos, only_exploding=True):
        x, y = pos
        for bomb_pos in self.bomb_positions:
            if only_exploding and not self._is_exploding(bomb_pos):
                continue
            bx, by = bomb_pos
            if pos == bomb_pos:
                return True
            if bx == x and abs(by - y) <= BLAST_RANGE and not self._blocked_line(bomb_pos, pos):
                return True
            if by == y and abs(bx - x) <= BLAST_RANGE and not self._blocked_line(bomb_pos, pos):
                return True
        return False

    def _danger_level(self, pos):
        level = 0
        x, y = pos
        for bomb_pos in self.bomb_positions:
            bx, by = bomb_pos
            in_line = False
            if bx == x and abs(by - y) <= BLAST_RANGE and not self._blocked_line(bomb_pos, pos):
                in_line = True
            if by == y and abs(bx - x) <= BLAST_RANGE and not self._blocked_line(bomb_pos, pos):
                in_line = True
            if pos == bomb_pos:
                in_line = True
            if in_line:
                if self.bomb_phases[bomb_pos] >= BOMB_PERIOD - 2:
                    return 2
                level = max(level, 1)
        return level

    def get_state(self):
        rx, ry = self.runner
        ex, ey = self.exit
        bx, by = self._nearest_bomb(self.runner)
        danger = self._danger_level(self.runner)
        dist_g = min(self._manhattan(self.runner, self.exit), 9)
        dist_b = min(self._nearest_bomb_dist(self.runner), 9)
        bomb_phase = max(self.bomb_phases.values())
        return (ex - rx, ey - ry, bx - rx, by - ry, danger, dist_g, dist_b, bomb_phase)

    def _move_runner(self, action):
        dx, dy = ACTIONS[action]
        x, y = self.runner
        nx = max(0, min(GRID_SIZE - 1, x + dx))
        ny = max(0, min(GRID_SIZE - 1, y + dy))
        if (nx, ny) not in self.walls and (nx, ny) not in self.bomb_positions:
            self.runner = (nx, ny)

    def _tick_bombs(self):
        for pos in self.bomb_positions:
            self.bomb_phases[pos] = (self.bomb_phases[pos] + 1) % BOMB_PERIOD

    def step(self, action):
        self.steps += 1
        old_pos = self.runner
        self._move_runner(action)
        moved = old_pos != self.runner
        self._tick_bombs()

        if self._in_blast(self.runner, only_exploding=True):
            return self.get_state(), -200.0, True, {"result": "exploded"}
        if self.runner == self.exit:
            return self.get_state(), 220.0, True, {"result": "escaped"}
        if self.steps >= MAX_STEPS:
            return self.get_state(), -50.0, True, {"result": "timeout"}

        reward = -0.3
        dist_g = self._manhattan(self.runner, self.exit)
        reward += (self._prev_dist_exit - dist_g) * 3.0
        self._prev_dist_exit = dist_g

        dist_b = self._nearest_bomb_dist(self.runner)
        reward += (dist_b - self._prev_danger_dist) * 1.2
        self._prev_danger_dist = dist_b

        danger = self._danger_level(self.runner)
        if danger == 2:
            reward -= 14.0
        elif danger == 1:
            reward -= 4.0

        if not moved and action != 4:
            reward -= 5.0
        if action == 4:
            reward -= 6.0 if danger > 0 else 0.5

        return self.get_state(), reward, False, {"result": "running"}


# ─────────────────────────────────────
# 웹 앱 상태 + 렌더링
# ─────────────────────────────────────
class WebGame:
    def __init__(self):
        self.env = BomberEscapeEnv()
        self.agent = QLearningAgent()
        self.episode = 0
        self.wins = 0
        self.losses = 0
        self.timeouts = 0
        self.recent_results = []
        self.demo_done = False
        self.last_result = "READY"
        self.last_action = 4
        self.step_count = 0
        for _ in range(PRETRAIN_EPISODES):
            self.train_one()
        self.start_new_demo()

    def train_one(self):
        state = self.env.reset()
        done = False
        info = {"result": "unknown"}
        while not done:
            action = self.agent.select_action(state)
            ns, reward, done, info = self.env.step(action)
            self.agent.learn(state, action, reward, ns, done)
            state = ns
        self.agent.decay_epsilon()
        self.episode += 1

        r = info["result"]
        if r == "escaped":
            self.wins += 1
            self.recent_results.append(1)
        elif r == "timeout":
            self.timeouts += 1
            self.recent_results.append(0)
        else:
            self.losses += 1
            self.recent_results.append(0)
        if len(self.recent_results) > 200:
            self.recent_results.pop(0)

    def train_many(self, n):
        for _ in range(int(n)):
            self.train_one()

    def reset_all(self):
        self.__init__()

    def start_new_demo(self):
        self.env.reset()
        self.demo_done = False
        self.last_result = "RUNNING"
        self.last_action = 4
        self.step_count = 0

    def demo_action(self, state):
        q = self.agent.get_q_values(state)
        if all(v == 0 for v in q):
            return random.randint(0, 4)
        if random.random() < 0.10:
            return random.randint(0, 4)
        return self.agent.best_action(state)

    def run_demo_step(self):
        if self.demo_done:
            return
        state = self.env.get_state()
        self.last_action = self.demo_action(state)
        _, _, done, info = self.env.step(self.last_action)
        self.demo_done = done
        self.last_result = info["result"].upper()
        self.step_count += 1

    def run_demo_episode(self):
        if self.demo_done:
            self.start_new_demo()
        while not self.demo_done:
            self.run_demo_step()

    def draw_cell_rect(self, draw, pos, fill, outline=None, width=1):
        x, y = pos
        rect = [x * CELL_SIZE, y * CELL_SIZE, (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE]
        draw.rectangle(rect, fill=fill, outline=outline, width=width)

    def draw_blast_range(self, draw):
        for bomb_pos in self.env.bomb_positions:
            bx, by = bomb_pos
            fill = (255, 90, 50, 95) if self.env._is_exploding(bomb_pos) else (255, 180, 60, 55)
            cells = [bomb_pos]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                for r in range(1, BLAST_RANGE + 1):
                    nx, ny = bx + dx * r, by + dy * r
                    if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                        break
                    if (nx, ny) in self.env.walls:
                        break
                    cells.append((nx, ny))
            for cell in cells:
                x, y = cell
                draw.rectangle([x * CELL_SIZE, y * CELL_SIZE, (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE], fill=fill)

    def draw_board(self, img):
        draw = ImageDraw.Draw(img, "RGBA")
        draw.rectangle([0, 0, BOARD_WIDTH, BOARD_WIDTH], fill=WHITE)

        self.draw_blast_range(draw)

        for x in range(GRID_SIZE + 1):
            draw.line([(x * CELL_SIZE, 0), (x * CELL_SIZE, BOARD_WIDTH)], fill=GRAY, width=1)
        for y in range(GRID_SIZE + 1):
            draw.line([(0, y * CELL_SIZE), (BOARD_WIDTH, y * CELL_SIZE)], fill=GRAY, width=1)

        for wall in self.env.walls:
            self.draw_cell_rect(draw, wall, DARK_GRAY, (110, 110, 110), 2)

        # exit
        ex, ey = self.env.exit
        cx = ex * CELL_SIZE + CELL_SIZE // 2
        cy = ey * CELL_SIZE + CELL_SIZE // 2
        draw.ellipse([cx - 21, cy - 21, cx + 21, cy + 21], fill=(90, 255, 150, 50))
        draw.rounded_rectangle([cx - 15, cy - 18, cx + 15, cy + 18], radius=5, fill=(50, 170, 90))
        draw.rounded_rectangle([cx - 10, cy - 13, cx + 10, cy + 13], radius=3, outline=(180, 255, 200), width=2)
        draw.ellipse([cx + 5, cy - 2, cx + 10, cy + 3], fill=(230, 255, 150))
        draw.text((cx - 16, cy + 21), "EXIT", font=FONT_Q, fill=(40, 140, 70))

        # bombs
        for bomb_pos in self.env.bomb_positions:
            bx, by = bomb_pos
            cx = bx * CELL_SIZE + CELL_SIZE // 2
            cy = by * CELL_SIZE + CELL_SIZE // 2
            phase = self.env.bomb_phases[bomb_pos]
            if self.env._is_exploding(bomb_pos):
                draw.ellipse([cx - 22, cy - 22, cx + 22, cy + 22], fill=(255, 170, 40))
                draw.ellipse([cx - 13, cy - 13, cx + 13, cy + 13], fill=(255, 70, 40))
            else:
                danger = phase / (BOMB_PERIOD - 1)
                color = (60 + int(170 * danger), 60, 60)
                draw.ellipse([cx - 15, cy - 15, cx + 15, cy + 15], fill=color)
                draw.ellipse([cx - 7, cy - 6, cx + 1, cy + 2], fill=(25, 25, 25))
                draw.line([(cx + 8, cy - 10), (cx + 17, cy - 20)], fill=ORANGE, width=3)
                draw.ellipse([cx + 15, cy - 26, cx + 23, cy - 18], fill=YELLOW)
                text = str(BOMB_PERIOD - 1 - phase)
                bbox = draw.textbbox((0, 0), text, font=FONT_Q)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text((cx - tw / 2, cy - th / 2), text, font=FONT_Q, fill=(255, 255, 255))

        # runner
        rx, ry = self.env.runner
        cx = rx * CELL_SIZE + CELL_SIZE // 2
        cy = ry * CELL_SIZE + CELL_SIZE // 2
        draw.ellipse([cx - 12, cy - 6, cx + 12, cy + 22], fill=BLUE)
        draw.ellipse([cx - 10, cy - 25, cx + 10, cy - 5], fill=(255, 220, 180))
        draw.arc([cx - 13, cy - 27, cx + 13, cy - 7], start=180, end=360, fill=(40, 80, 180), width=4)
        draw.ellipse([cx - 5, cy - 17, cx - 1, cy - 13], fill=(20, 20, 20))
        draw.ellipse([cx + 3, cy - 17, cx + 7, cy - 13], fill=(20, 20, 20))
        draw.line([(cx - 10, cy + 3), (cx - 18, cy + 10)], fill=(40, 80, 180), width=3)
        draw.line([(cx + 10, cy + 3), (cx + 18, cy + 10)], fill=(40, 80, 180), width=3)
        draw.line([(cx - 5, cy + 13), (cx - 8, cy + 23)], fill=(30, 60, 140), width=3)
        draw.line([(cx + 5, cy + 13), (cx + 8, cy + 23)], fill=(30, 60, 140), width=3)

        self.draw_action_arrow(draw)
        self.draw_q_overlay(draw)

    def draw_action_arrow(self, draw):
        if self.demo_done:
            return
        dx, dy = ACTIONS[self.last_action]
        if dx == 0 and dy == 0:
            return
        rx, ry = self.env.runner
        cx = rx * CELL_SIZE + CELL_SIZE // 2
        cy = ry * CELL_SIZE + CELL_SIZE // 2
        end = (cx + dx * 20, cy + dy * 20)
        draw.line([(cx, cy), end], fill=BLACK, width=3)
        draw.ellipse([end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4], fill=BLACK)

    def draw_q_overlay(self, draw):
        state = self.env.get_state()
        q = self.agent.get_q_values(state)
        rx, ry = self.env.runner
        cx = rx * CELL_SIZE + CELL_SIZE // 2
        cy = ry * CELL_SIZE + CELL_SIZE // 2
        offsets = {0: (0, -22), 1: (0, 22), 2: (-26, 0), 3: (26, 0), 4: (0, 0)}
        max_q = max(q)
        for a, (ox, oy) in offsets.items():
            v = q[a]
            color = (0, 150, 70) if v > 0 else (190, 30, 30) if v < 0 else (120, 120, 120)
            if abs(v - max_q) < 1e-9 and v != 0:
                draw.ellipse([cx + ox - 11, cy + oy - 11, cx + ox + 11, cy + oy + 11], outline=(0, 200, 100), width=2)
            text = f"{v:.1f}"
            draw.text((cx + ox - 12, cy + oy - 8), text, font=FONT_Q, fill=color)

    def draw_info_panel(self, img):
        draw = ImageDraw.Draw(img, "RGBA")
        py = BOARD_WIDTH
        pad = 10
        col = WIDTH // 3
        draw.rectangle([0, py, WIDTH, HEIGHT], fill=PANEL_BG)
        draw.line([(0, py), (WIDTH, py)], fill=(150, 150, 160), width=2)

        total = max(self.wins + self.losses + self.timeouts, 1)
        win_r = self.wins / total
        recent_r = sum(self.recent_results) / max(len(self.recent_results), 1)

        def txt(text, x, y, font=FONT_S, fill=(30, 30, 30)):
            draw.text((x, py + y), text, font=font, fill=fill)

        def bar(x, y, w, h, ratio, fg, bg=(205, 205, 205)):
            draw.rounded_rectangle([x, py + y, x + w, py + y + h], radius=3, fill=bg, outline=(160, 160, 160))
            fw = max(0, int(w * min(max(ratio, 0), 1)))
            if fw:
                draw.rounded_rectangle([x, py + y, x + fw, py + y + h], radius=3, fill=fg)

        row = 18
        x0 = pad
        txt("학습 통계", x0, 8, FONT_B, (50, 50, 80))
        txt(f"에피소드: {self.episode:,}", x0, 32)
        txt(f"탈출(승): {self.wins:,}", x0, 32 + row, fill=(40, 150, 80))
        txt(f"폭발(패): {self.losses:,}", x0, 32 + row * 2, fill=(190, 60, 60))
        txt(f"시간초과: {self.timeouts:,}", x0, 32 + row * 3, fill=(160, 120, 40))
        txt(f"전체 승률 {win_r * 100:.1f}%", x0, 32 + row * 4 + 4)
        bar(x0, 32 + row * 5 + 2, col - pad * 2, 10, win_r, (70, 190, 110) if win_r > 0.5 else (200, 100, 70))
        txt(f"최근200판 {recent_r * 100:.1f}%", x0, 32 + row * 6 + 6)
        bar(x0, 32 + row * 7 + 4, col - pad * 2, 10, recent_r, (70, 190, 110) if recent_r > 0.5 else (200, 100, 70))

        x1 = col + pad
        draw.line([(col, py + 6), (col, HEIGHT - 6)], fill=(190, 190, 200))
        txt("데모 상태", x1, 8, FONT_B, (50, 50, 80))
        result_color = {
            "ESCAPED": (40, 180, 90),
            "EXPLODED": (210, 60, 60),
            "TIMEOUT": (190, 140, 30),
            "RUNNING": (60, 120, 210),
            "READY": (120, 120, 120),
        }
        txt(self.last_result, x1, 32, FONT_B, result_color.get(self.last_result, BLACK))
        txt(f"스텝: {self.step_count} / {MAX_STEPS}", x1, 56)
        bar(x1, 72, col - pad * 2, 10, self.step_count / MAX_STEPS, (200, 80, 70) if self.step_count / MAX_STEPS > 0.7 else (70, 120, 210))
        txt(f"액션: {ACTION_NAMES[self.last_action]}", x1, 88)
        gdx, gdy, bdx, bdy, danger, dg, db, phase = self.env.get_state()
        txt(f"exit Δ({gdx:+d},{gdy:+d}) dist={dg}", x1, 108, fill=(80, 80, 80))
        txt(f"bomb Δ({bdx:+d},{bdy:+d}) dist={db}", x1, 126, fill=(80, 80, 80))
        d_color = (210, 60, 60) if danger == 2 else (190, 130, 30) if danger == 1 else (80, 160, 100)
        d_msg = "폭발위험" if danger == 2 else "주의" if danger == 1 else "안전"
        txt(f"폭발 위험: {d_msg} / phase={phase}", x1, 144, fill=d_color)

        x2 = col * 2 + pad
        draw.line([(col * 2, py + 6), (col * 2, HEIGHT - 6)], fill=(190, 190, 200))
        txt("에이전트", x2, 8, FONT_B, (50, 50, 80))
        eps = self.agent.epsilon
        eps_r = (eps - self.agent.epsilon_min) / (1.0 - self.agent.epsilon_min)
        txt(f"epsilon: {eps:.4f}", x2, 32)
        bar(x2, 48, col - pad * 2, 10, eps_r, (210, 150, 50), bg=(70, 180, 110))
        txt("왼쪽=학습완료 / 오른쪽=탐험중", x2, 62, fill=(130, 130, 130))
        txt(f"Q-table: {len(self.agent.q_table):,} 상태", x2, 82)
        txt(f"alpha={self.agent.alpha}  gamma={self.agent.gamma}", x2, 100)
        txt("버튼으로 학습/실행 가능", x2, 126, fill=(60, 90, 160))

    def render(self):
        img = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        self.draw_board(img)
        self.draw_info_panel(img)
        return img

    def markdown(self):
        total = max(self.wins + self.losses + self.timeouts, 1)
        recent = sum(self.recent_results) / max(len(self.recent_results), 1)
        return (
            f"### 현재 상태\n"
            f"- Episode: **{self.episode:,}**\n"
            f"- Win / Loss / Timeout: **{self.wins:,} / {self.losses:,} / {self.timeouts:,}**\n"
            f"- 전체 승률: **{self.wins / total * 100:.1f}%**, 최근 200판 승률: **{recent * 100:.1f}%**\n"
            f"- Demo result: **{self.last_result}**, step: **{self.step_count}/{MAX_STEPS}**, last action: **{ACTION_NAMES[self.last_action]}**\n"
            f"- Q-table states: **{len(self.agent.q_table):,}**, epsilon: **{self.agent.epsilon:.4f}**"
        )


def output(game):
    return game, game.render(), game.markdown()


def init_game():
    game = WebGame()
    return output(game)


def train(game, episodes):
    if game is None:
        game = WebGame()
    game.train_many(int(episodes))
    return output(game)


def demo_step(game):
    if game is None:
        game = WebGame()
    if game.demo_done:
        game.start_new_demo()
    else:
        game.run_demo_step()
    return output(game)


def demo_episode(game):
    if game is None:
        game = WebGame()
    game.run_demo_episode()
    return output(game)


def new_demo(game):
    if game is None:
        game = WebGame()
    game.start_new_demo()
    return output(game)


def reset_all():
    game = WebGame()
    return output(game)


with gr.Blocks(title="Bomber Escape RL") as demo:
    gr.Markdown(
        "# Bomber Escape RL - Gradio Web Version\n"
        "Pygame 창 대신 웹 버튼으로 Q-learning 에이전트를 학습하고 데모를 실행하는 버전입니다."
    )

    state = gr.State(None)
    with gr.Row():
        image = gr.Image(label="Game Screen", type="pil", height=760)
        info = gr.Markdown()

    with gr.Row():
        episodes = gr.Slider(10, 5000, value=500, step=10, label="추가 학습 episode 수")
        train_btn = gr.Button("선택한 만큼 학습")
        step_btn = gr.Button("데모 한 스텝")
        episode_btn = gr.Button("데모 한 판 끝까지")
        demo_btn = gr.Button("새 데모")
        reset_btn = gr.Button("전체 리셋")

    demo.load(init_game, inputs=None, outputs=[state, image, info])
    train_btn.click(train, inputs=[state, episodes], outputs=[state, image, info])
    step_btn.click(demo_step, inputs=state, outputs=[state, image, info])
    episode_btn.click(demo_episode, inputs=state, outputs=[state, image, info])
    demo_btn.click(new_demo, inputs=state, outputs=[state, image, info])
    reset_btn.click(reset_all, inputs=None, outputs=[state, image, info])

if __name__ == "__main__":

    demo.launch(server_name="127.0.0.1", server_port=7860)
