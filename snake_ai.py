"""
SNAKE NEON — RL EDITION (Fixed & Complete)
  [1] Play        — Human controlled
  [2] Train AI    — DQN with Target Network + Adam + Ray-cast state
  [3] Watch AI    — Watch trained model play
Requirements: pip install pygame numpy
"""
import pygame, random, math, sys, time, os
import numpy as np
from collections import deque

# ── Init ──────────────────────────────────────────────────────────────────────
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

W, H       = 1000, 720
CELL       = 26
COLS       = (W - 240) // CELL
ROWS       = (H - 90)  // CELL
BOARD_X    = 10
BOARD_Y    = 80
BOARD_W    = COLS * CELL
BOARD_H    = ROWS * CELL
MODEL_PATH = "snake_ai_model.npy"

screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("◈ SNAKE  NEON  —  RL EDITION ◈")
clock  = pygame.time.Clock()

# ── Palette ───────────────────────────────────────────────────────────────────
BG          = (4,   4,  14)
GRID_COL    = (18,  18,  42)
NEON_GREEN  = (57,  255,  20)
NEON_CYAN   = (0,   255, 230)
NEON_PINK   = (255,  20, 147)
NEON_YELLOW = (255, 230,   0)
NEON_PURPLE = (180,   0, 255)
NEON_ORANGE = (255, 100,   0)
NEON_BLUE   = ( 30, 120, 255)
WHITE       = (240, 240, 255)
DIM_WHITE   = (110, 110, 150)
RED_GLOW    = (255,  40,  40)
DARK_PANEL  = (  8,   8,  28)

# ── Fonts ─────────────────────────────────────────────────────────────────────
def _font(size, bold=False):
    for name in ("couriernew", "courier new", "monospace", "courier"):
        try:
            return pygame.font.SysFont(name, size, bold=bold)
        except Exception:
            pass
    return pygame.font.Font(None, size)

FONT_TITLE = _font(34, True)
FONT_HUD   = _font(21, True)
FONT_SMALL = _font(14)
FONT_BIG   = _font(68, True)
FONT_MED   = _font(26, True)
FONT_TINY  = _font(12)

# ── Sounds ────────────────────────────────────────────────────────────────────
def make_beep(freq=440, dur=0.07, vol=0.3, wave="square"):
    sr = 44100; n = int(sr * dur); buf = bytearray(n)
    for i in range(n):
        fade = min(1.0, (n - i) / (sr * 0.02))
        raw  = math.sin(2 * math.pi * freq * i / sr)
        if wave == "square": raw = 1.0 if raw >= 0 else -1.0
        buf[i] = int(127 + 127 * raw * vol * fade)
    return pygame.mixer.Sound(buffer=bytes(buf))

EAT_SOUND  = make_beep(880, 0.08, 0.35, "square")
DIE_SOUND  = make_beep(120, 0.40, 0.50, "square")
MOVE_SOUND = make_beep(330, 0.03, 0.10, "square")
SEL_SOUND  = make_beep(660, 0.05, 0.20, "square")

# ── Glow helpers ──────────────────────────────────────────────────────────────
def glow_rect(surf, color, rect, radius=12, steps=6):
    for i in range(steps, 0, -1):
        expand = i * (radius // steps)
        r = pygame.Rect(rect.x - expand, rect.y - expand,
                        rect.w + expand * 2, rect.h + expand * 2)
        s = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, max(0, 58 - i * 9)), (0, 0, r.w, r.h), border_radius=4)
        surf.blit(s, r.topleft)
    pygame.draw.rect(surf, color, rect, border_radius=2)

def glow_circle(surf, color, pos, rad, steps=5):
    for i in range(steps, 0, -1):
        expand = i * 3
        s = pygame.Surface(((rad + expand) * 2, (rad + expand) * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, max(0, 68 - i * 13)),
                           (rad + expand, rad + expand), rad + expand)
        surf.blit(s, (pos[0] - rad - expand, pos[1] - rad - expand))
    pygame.draw.circle(surf, color, pos, rad)

def glow_text(surf, font, text, color, pos, anchor="topleft"):
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        s = font.render(text, True, tuple(min(255, c // 2) for c in color))
        surf.blit(s, s.get_rect(**{anchor: (pos[0] + dx, pos[1] + dy)}))
    s = font.render(text, True, color)
    surf.blit(s, s.get_rect(**{anchor: pos}))

# ── Particles ─────────────────────────────────────────────────────────────────
class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color", "size")
    def __init__(self, x, y, color):
        a = random.uniform(0, math.tau); sp = random.uniform(1.5, 5.0)
        self.x, self.y   = float(x), float(y)
        self.vx, self.vy = math.cos(a) * sp, math.sin(a) * sp
        self.max_life = self.life = random.randint(20, 45)
        self.color = color; self.size = random.uniform(2, 5)
    def update(self):
        self.x += self.vx; self.y += self.vy
        self.vy += 0.12;   self.vx *= 0.96; self.life -= 1
    def draw(self, surf):
        r = self.life / self.max_life; sz = max(1, int(self.size * r))
        s = pygame.Surface((sz * 2 + 2, sz * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, int(255 * r)), (sz + 1, sz + 1), sz)
        surf.blit(s, (int(self.x) - sz - 1, int(self.y) - sz - 1))

particles: list = []

def spawn_particles(x, y, color, n=18):
    for _ in range(n): particles.append(Particle(x, y, color))

def tick_particles(surf):
    for p in particles[:]:
        p.update(); p.draw(surf)
        if p.life <= 0: particles.remove(p)

# ── Scanlines ─────────────────────────────────────────────────────────────────
_scanlines = pygame.Surface((W, H), pygame.SRCALPHA)
for _row in range(0, H, 4):
    pygame.draw.line(_scanlines, (0, 0, 0, 32), (0, _row), (W, _row))

# ── Game helpers ──────────────────────────────────────────────────────────────
FOOD_TYPES = [
    {"color": NEON_PINK,   "pts": 10, "label": "+10", "sz": 7},
    {"color": NEON_YELLOW, "pts": 20, "label": "+20", "sz": 6},
    {"color": NEON_CYAN,   "pts": 5,  "label": "+5",  "sz": 5},
    {"color": NEON_ORANGE, "pts": 30, "label": "+30", "sz": 8},
]

def new_food(snake):
    occ = set(map(tuple, snake))
    while True:
        p = (random.randint(0, COLS - 1), random.randint(0, ROWS - 1))
        if p not in occ:
            return p, random.randint(0, len(FOOD_TYPES) - 1)

def draw_grid(t):
    for x in range(BOARD_X, BOARD_X + BOARD_W + 1, CELL):
        pygame.draw.line(screen, GRID_COL, (x, BOARD_Y), (x, BOARD_Y + BOARD_H))
    for y in range(BOARD_Y, BOARD_Y + BOARD_H + 1, CELL):
        pygame.draw.line(screen, GRID_COL, (BOARD_X, y), (BOARD_X + BOARD_W, y))

def draw_head(surf, x, y, direction, color):
    cx = BOARD_X + x * CELL + CELL // 2
    cy = BOARD_Y + y * CELL + CELL // 2
    r  = pygame.Rect(BOARD_X + x * CELL + 3, BOARD_Y + y * CELL + 3, CELL - 6, CELL - 6)
    glow_rect(surf, color, r, radius=18, steps=7)
    offs = {(1, 0): [(4, -5), (4, 5)], (-1, 0): [(-4, -5), (-4, 5)],
            (0, 1): [(-5, 4), (5, 4)], (0, -1): [(-5, -4), (5, -4)]}
    for ex, ey in offs.get(direction, [(4, -5), (4, 5)]):
        pygame.draw.circle(surf, BG,    (cx + ex, cy + ey), 4)
        pygame.draw.circle(surf, WHITE, (cx + ex + 1, cy + ey - 1), 1)

def draw_body(surf, x, y, idx, length, color):
    ratio = 1 - idx / max(length, 1) * 0.4
    pad   = min(int(3 + idx * 0.3), CELL // 2 - 2)
    r = pygame.Rect(BOARD_X + x * CELL + pad, BOARD_Y + y * CELL + pad,
                    CELL - pad * 2, CELL - pad * 2)
    glow_rect(surf, tuple(int(c * ratio) for c in color), r, radius=10, steps=4)

def draw_food_item(surf, x, y, ftype, t):
    ft = FOOD_TYPES[ftype]
    cx = BOARD_X + x * CELL + CELL // 2
    cy = BOARD_Y + y * CELL + CELL // 2
    sz = ft["sz"] + int(2 * math.sin(t * 4))
    glow_circle(surf, ft["color"], (cx, cy), sz, steps=6)
    ox = int(cx + (sz + 5) * math.cos(t * 3))
    oy = int(cy + (sz + 5) * math.sin(t * 3))
    glow_circle(surf, WHITE, (ox, oy), 2, steps=3)

score_popups: list = []

def add_popup(x, y, text, color):
    score_popups.append({"x": float(BOARD_X + x * CELL + CELL // 2),
                         "y": float(BOARD_Y + y * CELL),
                         "text": text, "color": color, "life": 50})

def tick_popups(surf):
    for p in score_popups[:]:
        p["y"] -= 1.2; p["life"] -= 1
        s = FONT_HUD.render(p["text"], True, p["color"])
        s.set_alpha(min(255, p["life"] * 6))
        surf.blit(s, s.get_rect(centerx=int(p["x"]), centery=int(p["y"])))
        if p["life"] <= 0: score_popups.remove(p)

def draw_topbar(t, badge=""):
    pygame.draw.rect(screen, NEON_CYAN, (BOARD_X, BOARD_Y - 2, BOARD_W, BOARD_H + 4), 1)
    col = tuple(int(c * (0.7 + 0.3 * math.sin(t * 2))) for c in NEON_GREEN)
    glow_text(screen, FONT_TITLE, "◈ SNAKE  NEON ◈", col, (W // 2, 16), "midtop")
    if badge:
        bc = NEON_CYAN if "AI" in badge else NEON_YELLOW
        glow_text(screen, FONT_SMALL, badge, bc, (BOARD_X + BOARD_W // 2, 58), "midtop")
    if int(t * 2) % 2 == 0:
        for cx, cy in [(BOARD_X, BOARD_Y), (BOARD_X + BOARD_W, BOARD_Y),
                       (BOARD_X, BOARD_Y + BOARD_H), (BOARD_X + BOARD_W, BOARD_Y + BOARD_H)]:
            pygame.draw.rect(screen, NEON_CYAN, (cx - 3, cy - 3, 6, 6))

PANEL_X = BOARD_X + BOARD_W + 18
PANEL_Y = BOARD_Y

def _panel_base():
    pw = W - PANEL_X - 8
    s  = pygame.Surface((pw, BOARD_H), pygame.SRCALPHA)
    s.fill((8, 8, 28, 200)); screen.blit(s, (PANEL_X, PANEL_Y))
    pygame.draw.rect(screen, NEON_CYAN, (PANEL_X, PANEL_Y, pw, BOARD_H), 1, border_radius=4)
    return pw

def draw_play_panel(score, best, level, speed, length, t):
    pw = _panel_base()
    def sec(lbl, val, vy, col):
        glow_text(screen, FONT_SMALL, lbl, DIM_WHITE, (PANEL_X + 8, PANEL_Y + vy))
        glow_text(screen, FONT_HUD, str(val), col, (PANEL_X + pw // 2, PANEL_Y + vy + 16), "midtop")
    pulse = 0.6 + 0.4 * math.sin(t * 3)
    sec("SCORE",  score,        10,  tuple(int(c * pulse) for c in NEON_GREEN))
    sec("BEST",   best,         72,  NEON_YELLOW)
    sec("LEVEL",  level,        134, NEON_CYAN)
    sec("SPEED",  f"{speed}x",  196, NEON_PURPLE)
    sec("LENGTH", length,       258, NEON_ORANGE)
    hy = PANEL_Y + BOARD_H - 120
    glow_text(screen, FONT_SMALL, "── CONTROLS ──", DIM_WHITE, (PANEL_X + pw // 2, hy), "midtop")
    for i, line in enumerate(["WASD / ARROWS", "P = pause", "R = restart", "Q = menu"]):
        glow_text(screen, FONT_SMALL, line, DIM_WHITE, (PANEL_X + pw // 2, hy + 18 + i * 16), "midtop")

def draw_train_panel(agent, scores_hist, fast_mode, t):
    pw  = _panel_base()
    avg = sum(scores_hist[-50:]) / max(1, len(scores_hist[-50:])) if scores_hist else 0
    def sec(lbl, val, vy, col):
        glow_text(screen, FONT_SMALL, lbl, DIM_WHITE, (PANEL_X + 8, PANEL_Y + vy))
        glow_text(screen, FONT_HUD, str(val), col, (PANEL_X + pw // 2, PANEL_Y + vy + 16), "midtop")
    sec("EPISODE", agent.n_games,           10,  NEON_CYAN)
    sec("RECORD",  agent.record,             72,  NEON_YELLOW)
    sec("AVG-50",  f"{avg:.1f}",            134, NEON_GREEN)
    sec("ε-EXPL",  f"{agent.epsilon:.4f}",  196, NEON_ORANGE)
    bw = pw - 20; bx = PANEL_X + 10; by = PANEL_Y + 230
    pygame.draw.rect(screen, GRID_COL, (bx, by, bw, 10), border_radius=3)
    filled = int(bw * agent.epsilon)
    if filled > 0:
        pygame.draw.rect(screen, NEON_ORANGE, (bx, by, filled, 10), border_radius=3)
    mode_col  = NEON_PINK if fast_mode else NEON_BLUE
    mode_text = "FAST  MODE" if fast_mode else "VISUAL MODE"
    glow_text(screen, FONT_SMALL, "SPACE = toggle speed", DIM_WHITE,
              (PANEL_X + pw // 2, PANEL_Y + 250), "midtop")
    glow_text(screen, FONT_HUD, mode_text, mode_col,
              (PANEL_X + pw // 2, PANEL_Y + 268), "midtop")
    if len(scores_hist) > 1:
        glow_text(screen, FONT_SMALL, "── SCORE GRAPH ──", DIM_WHITE,
                  (PANEL_X + pw // 2, PANEL_Y + 300), "midtop")
        gx, gy, gw, gh = PANEL_X + 10, PANEL_Y + 318, pw - 20, 90
        pygame.draw.rect(screen, GRID_COL, (gx, gy, gw, gh), border_radius=2)
        hist = scores_hist[-80:]; mx = max(max(hist), 1); pts = []
        for i, v in enumerate(hist):
            px = gx + int(i / (len(hist) - 1) * gw) if len(hist) > 1 else gx + gw // 2
            py = gy + gh - int(v / mx * gh); pts.append((px, py))
        if len(pts) > 1:
            pygame.draw.lines(screen, NEON_GREEN, False, pts, 2)
        pygame.draw.circle(screen, NEON_YELLOW, pts[-1], 3)
    hy = PANEL_Y + BOARD_H - 80
    glow_text(screen, FONT_SMALL, "── CONTROLS ──", DIM_WHITE, (PANEL_X + pw // 2, hy), "midtop")
    for i, line in enumerate(["SPACE = fast/visual", "S = save", "ESC = menu"]):
        glow_text(screen, FONT_SMALL, line, DIM_WHITE, (PANEL_X + pw // 2, hy + 16 + i * 14), "midtop")

def draw_watch_panel(score, best, episode, length, t):
    pw = _panel_base()
    def sec(lbl, val, vy, col):
        glow_text(screen, FONT_SMALL, lbl, DIM_WHITE, (PANEL_X + 8, PANEL_Y + vy))
        glow_text(screen, FONT_HUD, str(val), col, (PANEL_X + pw // 2, PANEL_Y + vy + 16), "midtop")
    pulse = 0.6 + 0.4 * math.sin(t * 3)
    sec("SCORE",   score,   10,  tuple(int(c * pulse) for c in NEON_CYAN))
    sec("BEST",    best,    72,  NEON_YELLOW)
    sec("EPISODE", episode, 134, NEON_PURPLE)
    sec("LENGTH",  length,  196, NEON_GREEN)
    by = PANEL_Y + 250
    glow_text(screen, FONT_SMALL, "── AI PLAYING ──", NEON_CYAN, (PANEL_X + pw // 2, by), "midtop")
    brain_y = by + 24
    for i in range(5):
        h  = int(10 + 8 * math.sin(t * 4 + i * 0.8))
        bx = PANEL_X + 12 + i * (pw - 20) // 4
        pygame.draw.rect(screen, NEON_CYAN, (bx, brain_y + 20 - h, 8, h), border_radius=2)
    glow_text(screen, FONT_TINY, "32-input ray-cast brain", DIM_WHITE,
              (PANEL_X + pw // 2, brain_y + 36), "midtop")
    hy = PANEL_Y + BOARD_H - 80
    glow_text(screen, FONT_SMALL, "── CONTROLS ──", DIM_WHITE, (PANEL_X + pw // 2, hy), "midtop")
    for i, line in enumerate(["R = restart", "ESC = menu", "Q = quit"]):
        glow_text(screen, FONT_SMALL, line, DIM_WHITE, (PANEL_X + pw // 2, hy + 16 + i * 14), "midtop")

def draw_overlay(lines, t):
    s = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
    s.fill((4, 4, 18, 215)); screen.blit(s, (BOARD_X, BOARD_Y))
    pygame.draw.rect(screen, NEON_CYAN, (BOARD_X, BOARD_Y, BOARD_W, BOARD_H), 2)
    mid_x = BOARD_X + BOARD_W // 2; mid_y = BOARD_Y + BOARD_H // 2
    total = len(lines) * 52
    for i, (text, font, color) in enumerate(lines):
        c = tuple(int(ch * (0.7 + 0.3 * math.sin(t * 2.5 + i))) for ch in color)
        glow_text(screen, font, text, c, (mid_x, mid_y - total // 2 + i * 52), "midtop")

# ─────────────────────────────────────────────────────────────────────────────
# RAY-CAST STATE  — 32 features
# 8 directions × 3 values (wall, body, food) = 24
# + 4 one-hot direction + 4 food relative = 32
# ─────────────────────────────────────────────────────────────────────────────
DIRS_8 = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]

def build_state(snake, direction, food):
    occ   = set(map(tuple, snake[1:]))   # body only, not head
    hx, hy = snake[0]; fx, fy = food
    max_d  = max(COLS, ROWS)
    ray_feats = []
    for ddx, ddy in DIRS_8:
        cx, cy   = hx, hy
        wall_sig = 0.0; body_sig = 0.0; food_hit = 0.0
        for step in range(1, max_d + 1):
            cx += ddx; cy += ddy
            if cx < 0 or cx >= COLS or cy < 0 or cy >= ROWS:
                wall_sig = 1.0 / step   # sharp: strong when close
                break
            if (cx, cy) in occ and body_sig == 0.0:
                body_sig = 1.0 / step   # sharp: strong when close
            if (cx, cy) == (fx, fy):
                food_hit = 1.0          # no break: wall still detected behind food
        ray_feats.extend([wall_sig, body_sig, food_hit])
    dir_oh   = [float(direction == (-1, 0)), float(direction == (1, 0)),
                float(direction == (0, -1)), float(direction == (0,  1))]
    food_rel = [float(fx < hx), float(fx > hx), float(fy < hy), float(fy > hy)]
    return np.array(ray_feats + dir_oh + food_rel, dtype=np.float32)  # shape (32,)

STATE_SIZE = 32

# ─────────────────────────────────────────────────────────────────────────────
# NEURAL NETWORK  32 → 256 → 256 → 3  with Adam optimizer
# ─────────────────────────────────────────────────────────────────────────────
class QNet:
    LR = 1e-3; BETA1 = 0.9; BETA2 = 0.999; EPS = 1e-8

    def __init__(self):
        s1 = np.sqrt(2 / STATE_SIZE); s2 = np.sqrt(2 / 256)   # correct Xavier
        self.W1 = np.random.randn(STATE_SIZE, 256) * s1; self.b1 = np.zeros(256)
        self.W2 = np.random.randn(256, 256) * s2;        self.b2 = np.zeros(256)
        self.W3 = np.random.randn(256, 3) * s2;          self.b3 = np.zeros(3)
        self._t = 0
        self._m = [np.zeros_like(p) for p in self._params()]
        self._v = [np.zeros_like(p) for p in self._params()]

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x):
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3, h1, h2

    def predict(self, x):
        return self.forward(x.reshape(1, -1))[0][0]

    def train(self, states, targets):       # Adam — no external LR needed
        out, h1, h2 = self.forward(states)
        dL  = 2 * (out - targets) / len(states)
        dW3 = h2.T @ dL;    db3 = dL.sum(0)
        dh2 = (dL @ self.W3.T) * (h2 > 0)
        dW2 = h1.T @ dh2;   db2 = dh2.sum(0)
        dh1 = (dh2 @ self.W2.T) * (h1 > 0)
        dW1 = states.T @ dh1; db1 = dh1.sum(0)
        grads = [dW1, db1, dW2, db2, dW3, db3]
        self._t += 1
        lr_t = self.LR * (1 - self.BETA2 ** self._t) ** 0.5 / (1 - self.BETA1 ** self._t)
        for i, (p, g) in enumerate(zip(self._params(), grads)):
            self._m[i] = self.BETA1 * self._m[i] + (1 - self.BETA1) * g
            self._v[i] = self.BETA2 * self._v[i] + (1 - self.BETA2) * g * g
            p -= lr_t * self._m[i] / (np.sqrt(self._v[i]) + self.EPS)

    def save(self):
        np.save(MODEL_PATH, {"W1": self.W1, "b1": self.b1,
                             "W2": self.W2, "b2": self.b2,
                             "W3": self.W3, "b3": self.b3,
                             "t":  self._t,  "m":  self._m, "v": self._v})
        print(f"[AI] Model saved → {MODEL_PATH}")

    def load(self):
        if not os.path.exists(MODEL_PATH): return False
        d = np.load(MODEL_PATH, allow_pickle=True).item()
        # shape check — reject incompatible old models
        if d["W1"].shape != (STATE_SIZE, 256):
            print("[AI] Incompatible model — starting fresh")
            return False
        self.W1, self.b1 = d["W1"], d["b1"]
        self.W2, self.b2 = d["W2"], d["b2"]
        self.W3, self.b3 = d["W3"], d["b3"]
        if "t" in d:
            self._t = int(d["t"]); self._m = list(d["m"]); self._v = list(d["v"])
        return True

# ─────────────────────────────────────────────────────────────────────────────
# SNAKE ENVIRONMENT  (headless, used during training)
# ─────────────────────────────────────────────────────────────────────────────
class SnakeEnv:
    def reset(self):
        self.snake = [(COLS//2, ROWS//2), (COLS//2-1, ROWS//2), (COLS//2-2, ROWS//2)]
        self.dir   = (1, 0)
        self.food, self.ftype = new_food(self.snake)
        self.score = 0; self.hunger = 0
        return build_state(self.snake, self.dir, self.food)   # 32 features

    def step(self, action):
        dx, dy = self.dir
        if   action == 1: self.dir = (-dy,  dx)   # turn right
        elif action == 2: self.dir = ( dy, -dx)   # turn left
        self.hunger += 1
        hx, hy = self.snake[0]
        head   = (hx + self.dir[0], hy + self.dir[1])
        occ    = set(map(tuple, self.snake))

        if (head[0] < 0 or head[0] >= COLS or
                head[1] < 0 or head[1] >= ROWS or
                head in occ or
                self.hunger > max(100 * len(self.snake), 200 * COLS)):
            return build_state(self.snake, self.dir, self.food), -10.0, True

        self.snake.insert(0, head)

        if head == self.food:
            self.score  += 1
            self.hunger  = 0
            self.food, self.ftype = new_food(self.snake)
            return build_state(self.snake, self.dir, self.food), 10.0, False
        else:
            self.snake.pop()
            fx, fy     = self.food
            old_dist   = abs(hx - fx) + abs(hy - fy)
            new_dist   = abs(head[0] - fx) + abs(head[1] - fy)
            length_scale = max(0.3, 1.0 - len(self.snake) / (COLS * ROWS))
            reward     = 0.1 * length_scale if new_dist < old_dist else -0.1 * length_scale
            if len(self.snake) > 20:
                reward += 0.01 * (len(self.snake) / (COLS * ROWS))
            return build_state(self.snake, self.dir, self.food), reward, False

# ─────────────────────────────────────────────────────────────────────────────
# DQN AGENT  with Target Network
# ─────────────────────────────────────────────────────────────────────────────
class Agent:
    GAMMA = 0.95; BATCH = 512; MEM = 100_000
    EPS_MIN = 0.01; EPS_DECAY = 0.995
    TARGET_UPDATE_FREQ = 500

    def __init__(self):
        self.model        = QNet()
        self.target_model = QNet()
        self._copy_weights()
        self.memory     = deque(maxlen=self.MEM)
        self.epsilon    = 1.0
        self.n_games    = 0
        self.record     = 0
        self.scores     = []
        self.step_count = 0

    def _copy_weights(self):
        self.target_model.W1 = self.model.W1.copy()
        self.target_model.b1 = self.model.b1.copy()
        self.target_model.W2 = self.model.W2.copy()
        self.target_model.b2 = self.model.b2.copy()
        self.target_model.W3 = self.model.W3.copy()
        self.target_model.b3 = self.model.b3.copy()

    def act(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, 2)
        return int(np.argmax(self.model.predict(state)))

    def remember(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))

    def _train_batch(self, batch):
        s, a, r, ns, d = zip(*batch)
        s  = np.array(s,  dtype=np.float32)
        ns = np.array(ns, dtype=np.float32)
        r  = np.array(r,  dtype=np.float32)
        d  = np.array(d,  dtype=bool)
        curr_q, _, _ = self.model.forward(s)
        next_q, _, _ = self.target_model.forward(ns)   # frozen target network
        targets = curr_q.copy()
        for i, ai in enumerate(a):
            targets[i, ai] = r[i] if d[i] else r[i] + self.GAMMA * float(np.max(next_q[i]))
        self.model.train(s, targets)   # Adam handles LR internally

    def train_short(self, s, a, r, ns, done):
        self._train_batch([(s, a, r, ns, done)])
        self.step_count += 1
        if self.step_count % self.TARGET_UPDATE_FREQ == 0:
            self._copy_weights()

    def train_long(self):
        batch = (random.sample(list(self.memory), self.BATCH)
                 if len(self.memory) >= self.BATCH else list(self.memory))
        if batch: self._train_batch(batch)

    def end_episode(self, score):
        self.n_games += 1; self.scores.append(score)
        if score > self.record: self.record = score
        self.epsilon = max(self.EPS_MIN, self.epsilon * self.EPS_DECAY)
        if self.n_games % 2000 == 0 and self.n_games > 0:
            self.epsilon = min(0.15, self.epsilon + 0.1)
            print(f"[AI] Epsilon nudged to {self.epsilon:.3f} at episode {self.n_games}")
        self.train_long()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────────────────────────────────────────
def main_menu():
    model_exists = os.path.exists(MODEL_PATH)
    sel = 0
    options = [
        ("PLAY",     "Human controlled — classic snake",      NEON_GREEN),
        ("TRAIN AI", "DQN + Target Network + Ray-cast state", NEON_CYAN),
        ("WATCH AI", "Watch the trained AI play",             NEON_PURPLE),
        ("QUIT",     "",                                       RED_GLOW),
    ]
    t = 0.0
    bsnake = [(COLS // 2 + i, ROWS // 2) for i in range(8)]
    bdir = (1, 0); bfood, bft = new_food(bsnake); bfood_t = 0.0
    last_bt = time.time()

    while True:
        dt = clock.tick(60) / 1000.0; t += dt; bfood_t += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_UP,    pygame.K_w): sel = (sel - 1) % len(options); SEL_SOUND.play()
                if k in (pygame.K_DOWN,  pygame.K_s): sel = (sel + 1) % len(options); SEL_SOUND.play()
                if k in (pygame.K_RETURN, pygame.K_SPACE):
                    EAT_SOUND.play(); lbl = options[sel][0]
                    if lbl == "PLAY":     return "play"
                    if lbl == "TRAIN AI": return "train"
                    if lbl == "WATCH AI": return "watch"
                    if lbl == "QUIT":     pygame.quit(); sys.exit()
                if k == pygame.K_1: return "play"
                if k == pygame.K_2: return "train"
                if k == pygame.K_3: return "watch"
                if k == pygame.K_q: pygame.quit(); sys.exit()

        now = time.time()
        if now - last_bt > 0.12:
            last_bt = now; dx, dy = bdir; hx, hy = bsnake[0]; fx, fy = bfood
            best_m = bdir; best_d = 9999
            for m in [(1,0),(-1,0),(0,1),(0,-1)]:
                if m == (-dx, -dy): continue
                nx, ny = (hx + m[0]) % COLS, (hy + m[1]) % ROWS
                if (nx, ny) not in bsnake:
                    d2 = abs(nx - fx) + abs(ny - fy)
                    if d2 < best_d: best_d = d2; best_m = m
            bdir = best_m
            head = ((bsnake[0][0] + bdir[0]) % COLS, (bsnake[0][1] + bdir[1]) % ROWS)
            if head not in bsnake:
                bsnake.insert(0, head)
                if head == bfood: bfood, bft = new_food(bsnake)
                else:             bsnake.pop()

        screen.fill(BG); draw_grid(t)
        for sx, sy in bsnake:
            pad = 4
            r   = pygame.Rect(BOARD_X+sx*CELL+pad, BOARD_Y+sy*CELL+pad, CELL-pad*2, CELL-pad*2)
            s   = pygame.Surface((r.w, r.h), pygame.SRCALPHA); s.fill((*NEON_GREEN, 28))
            screen.blit(s, r.topleft)
        draw_food_item(screen, bfood[0], bfood[1], bft, bfood_t * 0.5)
        vig = pygame.Surface((W, H), pygame.SRCALPHA); vig.fill((4, 4, 14, 160))
        screen.blit(vig, (0, 0))
        pygame.draw.rect(screen, NEON_CYAN, (BOARD_X, BOARD_Y, BOARD_W, BOARD_H), 1)
        pulse = 0.75 + 0.25 * math.sin(t * 1.8)
        glow_text(screen, FONT_BIG, "SNAKE",
                  tuple(int(c * pulse) for c in NEON_GREEN), (W // 2, 30), "midtop")
        glow_text(screen, FONT_MED, "NEON  ✦  RL EDITION", NEON_CYAN, (W // 2, 108), "midtop")
        box_w, box_h = 480, 54; bx = W // 2 - box_w // 2
        for i, (lbl, desc, col) in enumerate(options):
            by2    = 162 + i * 68; is_sel = (i == sel)
            br     = pygame.Rect(bx, by2, box_w, box_h)
            if is_sel:
                glow_rect(screen, col, br, radius=14, steps=5)
                pygame.draw.rect(screen, col, br, 2, border_radius=4)
            else:
                s = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
                s.fill((14, 14, 40, 180)); screen.blit(s, (bx, by2))
                pygame.draw.rect(screen, DIM_WHITE, (bx, by2, box_w, box_h), 1, border_radius=4)
            tc = col if is_sel else DIM_WHITE
            glow_text(screen, FONT_HUD, f"[{i+1}]  {lbl}", tc, (bx + 20, by2 + 8))
            if desc and is_sel: glow_text(screen, FONT_TINY, desc, WHITE, (bx + 20, by2 + 34))
        ms_col  = NEON_GREEN if model_exists else NEON_ORANGE
        ms_text = f"✓ MODEL FOUND  ({MODEL_PATH})" if model_exists else "✗ NO MODEL — train first!"
        glow_text(screen, FONT_SMALL, ms_text, ms_col, (W // 2, 490), "midtop")
        glow_text(screen, FONT_TINY, "↑ ↓  NAVIGATE    ENTER  SELECT    Q  QUIT",
                  DIM_WHITE, (W // 2, 510), "midtop")
        screen.blit(_scanlines, (0, 0)); pygame.display.flip()

# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — HUMAN PLAY
# ─────────────────────────────────────────────────────────────────────────────
def game():
    snake     = [(COLS//2, ROWS//2), (COLS//2-1, ROWS//2), (COLS//2-2, ROWS//2)]
    direction = (1, 0); next_dir = (1, 0)
    food, ftype = new_food(snake)
    score = 0; best = 0; level = 1; base_fps = 9; fps = base_fps; speed_disp = 1.0
    paused = False; dead = False; started = False; flash = 0
    t = 0.0; last_move = time.time(); death_anim = 0

    while True:
        dt = clock.tick(60) / 1000.0; t += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                k = event.key
                if k == pygame.K_q: return "menu"
                if k == pygame.K_r: return "play"
                if not dead:
                    if k == pygame.K_p: paused = not paused
                    _mv = {pygame.K_UP: (0,-1), pygame.K_w: (0,-1),
                           pygame.K_DOWN: (0,1), pygame.K_s: (0,1),
                           pygame.K_LEFT: (-1,0), pygame.K_a: (-1,0),
                           pygame.K_RIGHT: (1,0), pygame.K_d: (1,0)}
                    mv = _mv.get(k)
                    if mv and (mv[0] != -direction[0] or mv[1] != -direction[1]):
                        next_dir = mv; started = True

        if not paused and not dead and started:
            now = time.time()
            if now - last_move >= 1 / fps:
                last_move = now; direction = next_dir
                head = (snake[0][0] + direction[0], snake[0][1] + direction[1])
                if not (0 <= head[0] < COLS and 0 <= head[1] < ROWS) or head in snake:
                    dead = True; DIE_SOUND.play()
                    spawn_particles(BOARD_X+snake[0][0]*CELL+CELL//2,
                                    BOARD_Y+snake[0][1]*CELL+CELL//2, RED_GLOW, 40)
                    continue
                snake.insert(0, head); MOVE_SOUND.play()
                if head == food:
                    ft = FOOD_TYPES[ftype]; score += ft["pts"]; best = max(best, score)
                    EAT_SOUND.play(); flash = 8
                    spawn_particles(BOARD_X+food[0]*CELL+CELL//2,
                                    BOARD_Y+food[1]*CELL+CELL//2, ft["color"], 30)
                    add_popup(food[0], food[1], ft["label"], ft["color"])
                    food, ftype = new_food(snake)
                    level = 1 + score // 50; fps = base_fps + (level - 1) * 1.2
                    speed_disp = round(fps / base_fps, 1)
                else: snake.pop()
        if dead: death_anim += 1

        screen.fill(BG)
        if flash > 0:
            fs = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
            fs.fill((255, 255, 255, flash * 8)); screen.blit(fs, (BOARD_X, BOARD_Y)); flash -= 1
        draw_grid(t); draw_topbar(t, "▶  PLAYER MODE")
        tick_particles(screen); draw_food_item(screen, food[0], food[1], ftype, t)
        if not dead:
            for i, (sx, sy) in enumerate(reversed(snake[1:])):
                draw_body(screen, sx, sy, len(snake)-1-i, len(snake), NEON_GREEN)
            if snake: draw_head(screen, snake[0][0], snake[0][1], direction, NEON_GREEN)
        else:
            col = RED_GLOW if death_anim % 6 < 3 else NEON_ORANGE
            for sx, sy in snake:
                if random.random() > death_anim * 0.03:
                    glow_rect(screen, col,
                              pygame.Rect(BOARD_X+sx*CELL+3, BOARD_Y+sy*CELL+3, CELL-6, CELL-6), 8, 3)
        tick_popups(screen)
        draw_play_panel(score, best, level, speed_disp, len(snake), t)
        screen.blit(_scanlines, (0, 0))
        if not started and not dead:
            draw_overlay([("SNAKE", FONT_BIG, NEON_GREEN), ("NEON EDITION", FONT_MED, NEON_CYAN),
                          ("", FONT_SMALL, NEON_CYAN), ("WASD / ARROWS", FONT_HUD, DIM_WHITE),
                          ("TO  BEGIN", FONT_HUD, NEON_YELLOW)], t)
        elif paused:
            draw_overlay([("PAUSED", FONT_BIG, NEON_YELLOW), ("P  resume", FONT_HUD, DIM_WHITE),
                          ("R  restart", FONT_HUD, DIM_WHITE)], t)
        elif dead and death_anim > 30:
            draw_overlay([("GAME OVER", FONT_BIG, RED_GLOW),
                          (f"SCORE: {score}", FONT_MED, NEON_YELLOW),
                          (f"BEST:  {best}",  FONT_MED, NEON_GREEN),
                          ("R  restart", FONT_HUD, DIM_WHITE),
                          ("Q  menu",    FONT_HUD, DIM_WHITE)], t)
        pygame.display.flip()

# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — TRAIN AI
# ─────────────────────────────────────────────────────────────────────────────
def train_ai():
    agent = Agent(); agent.model.load()
    env   = SnakeEnv(); state = env.reset()
    fast_mode = False; t = 0.0
    FAST_SPF  = 200
    notify = ""; notify_timer = 0.0

    while True:
        dt = clock.tick(60) / 1000.0; t += dt
        if notify_timer > 0: notify_timer -= dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return "menu"
                if event.key == pygame.K_SPACE:  fast_mode = not fast_mode; SEL_SOUND.play()
                if event.key == pygame.K_s:
                    agent.model.save(); notify = "✓ MODEL SAVED"
                    notify_timer = 2.5; EAT_SOUND.play()

        steps = FAST_SPF if fast_mode else 1
        for _ in range(steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train_short(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.end_episode(env.score)
                if agent.n_games % 500 == 0:
                    agent.model.save()
                    notify = f"✓ AUTO-SAVED (ep {agent.n_games})"; notify_timer = 2.5
                state = env.reset()

        screen.fill(BG)
        if fast_mode:
            draw_grid(t); draw_topbar(t, "⚡ FAST  TRAINING")
            avg = sum(agent.scores[-50:]) / max(1, len(agent.scores[-50:])) if agent.scores else 0
            glow_text(screen, FONT_MED, f"Episode  {agent.n_games}", NEON_CYAN,
                      (BOARD_X+BOARD_W//2, BOARD_Y+BOARD_H//2-60), "midtop")
            glow_text(screen, FONT_MED, f"Record   {agent.record}", NEON_YELLOW,
                      (BOARD_X+BOARD_W//2, BOARD_Y+BOARD_H//2-10), "midtop")
            glow_text(screen, FONT_HUD, f"ε = {agent.epsilon:.5f}", NEON_ORANGE,
                      (BOARD_X+BOARD_W//2, BOARD_Y+BOARD_H//2+38), "midtop")
            glow_text(screen, FONT_HUD, f"avg-50 = {avg:.1f}", NEON_GREEN,
                      (BOARD_X+BOARD_W//2, BOARD_Y+BOARD_H//2+78), "midtop")
            pulse_col = tuple(int(c * (0.4 + 0.6 * abs(math.sin(t * 4)))) for c in NEON_PINK)
            pygame.draw.rect(screen, pulse_col, (BOARD_X, BOARD_Y, BOARD_W, BOARD_H), 2)
        else:
            draw_grid(t); draw_topbar(t, "◉ TRAINING  MODE")
            draw_food_item(screen, env.food[0], env.food[1], env.ftype, t)
            for i, (sx, sy) in enumerate(reversed(env.snake[1:])):
                draw_body(screen, sx, sy, len(env.snake)-1-i, len(env.snake), NEON_CYAN)
            if env.snake:
                draw_head(screen, env.snake[0][0], env.snake[0][1], env.dir, NEON_CYAN)

        draw_train_panel(agent, agent.scores, fast_mode, t)
        if notify and notify_timer > 0:
            glow_text(screen, FONT_HUD, notify, NEON_GREEN,
                      (BOARD_X+BOARD_W//2, BOARD_Y+BOARD_H-50), "midtop")
        screen.blit(_scanlines, (0, 0)); pygame.display.flip()

# ─────────────────────────────────────────────────────────────────────────────
# MODE 3 — WATCH AI
# ─────────────────────────────────────────────────────────────────────────────
def watch_ai():
    agent = Agent()
    if not agent.model.load():
        for _ in range(180):
            screen.fill(BG); draw_grid(0)
            draw_overlay([("NO MODEL FOUND",    FONT_MED, RED_GLOW),
                          ("Train the AI first!", FONT_HUD, NEON_ORANGE),
                          ("ESC / Q = menu",      FONT_HUD, DIM_WHITE)], 0)
            screen.blit(_scanlines, (0, 0)); pygame.display.flip(); clock.tick(60)
            for e in pygame.event.get():
                if e.type == pygame.QUIT: pygame.quit(); sys.exit()
                if e.type == pygame.KEYDOWN: return "menu"
        return "menu"

    agent.epsilon = 0.0   # pure greedy — no exploration
    snake     = [(COLS//2, ROWS//2), (COLS//2-1, ROWS//2), (COLS//2-2, ROWS//2)]
    direction = (1, 0)
    food, ftype = new_food(snake)
    score = 0; best = 0; episode = 1; flash = 0
    dead  = False; death_anim = 0; hunger = 0
    t = 0.0; last_move = time.time(); fps = 10

    while True:
        dt = clock.tick(60) / 1000.0; t += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q): return "menu"
                if event.key == pygame.K_r:
                    snake     = [(COLS//2, ROWS//2), (COLS//2-1, ROWS//2), (COLS//2-2, ROWS//2)]
                    direction = (1, 0); food, ftype = new_food(snake)
                    score = 0; dead = False; death_anim = 0; hunger = 0; flash = 0; episode += 1

        if not dead:
            now = time.time()
            if now - last_move >= 1 / fps:
                last_move = now
                # use build_state — same 32-feature state used in training
                state  = build_state(snake, direction, food)
                action = agent.act(state, greedy=True)
                dx, dy = direction
                if   action == 1: direction = (-dy,  dx)
                elif action == 2: direction = ( dy, -dx)
                head   = (snake[0][0] + direction[0], snake[0][1] + direction[1])
                hunger += 1
                if (not (0 <= head[0] < COLS and 0 <= head[1] < ROWS)
                        or head in snake
                        or hunger > max(100 * len(snake), 200 * COLS)):
                    dead = True; DIE_SOUND.play(); death_anim = 0
                    spawn_particles(BOARD_X+snake[0][0]*CELL+CELL//2,
                                    BOARD_Y+snake[0][1]*CELL+CELL//2, RED_GLOW, 40)
                else:
                    snake.insert(0, head)
                    if head == food:
                        ft = FOOD_TYPES[ftype]; score += 1; best = max(best, score); hunger = 0
                        EAT_SOUND.play(); flash = 8
                        spawn_particles(BOARD_X+food[0]*CELL+CELL//2,
                                        BOARD_Y+food[1]*CELL+CELL//2, ft["color"], 30)
                        add_popup(food[0], food[1], ft["label"], ft["color"])
                        food, ftype = new_food(snake)
                    else: snake.pop()
        else:
            death_anim += 1
            if death_anim > 80:
                snake     = [(COLS//2, ROWS//2), (COLS//2-1, ROWS//2), (COLS//2-2, ROWS//2)]
                direction = (1, 0); food, ftype = new_food(snake)
                score = 0; dead = False; death_anim = 0; hunger = 0; flash = 0; episode += 1

        screen.fill(BG)
        if flash > 0:
            fs = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
            fs.fill((255, 255, 255, flash * 8)); screen.blit(fs, (BOARD_X, BOARD_Y)); flash -= 1
        draw_grid(t); draw_topbar(t, "🤖 AI  CONTROLLED")
        tick_particles(screen); draw_food_item(screen, food[0], food[1], ftype, t)
        if not dead:
            for i, (sx, sy) in enumerate(reversed(snake[1:])):
                draw_body(screen, sx, sy, len(snake)-1-i, len(snake), NEON_PURPLE)
            if snake: draw_head(screen, snake[0][0], snake[0][1], direction, NEON_PURPLE)
        else:
            col = RED_GLOW if death_anim % 6 < 3 else NEON_ORANGE
            for sx, sy in snake:
                if random.random() > death_anim * 0.03:
                    glow_rect(screen, col,
                              pygame.Rect(BOARD_X+sx*CELL+3, BOARD_Y+sy*CELL+3, CELL-6, CELL-6), 8, 3)
        tick_popups(screen)
        draw_watch_panel(score, best, episode, len(snake), t)
        screen.blit(_scanlines, (0, 0)); pygame.display.flip()

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    state = "menu"
    while True:
        if   state == "menu":  state = main_menu()
        elif state == "play":  state = game()
        elif state == "train": state = train_ai()
        elif state == "watch": state = watch_ai()
        else:                  state = "menu"