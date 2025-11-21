# environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import List, Dict, Tuple

# --- Grid Encoding ---
CELL_LAVA = -100
CELL_EMPTY = 0
CELL_BARRIER = 100
# Boxes are positive ints 1..10 (we use 10 as a generic box value)

# --- Actions ---
ACTION_MOVE_UP = 0
ACTION_MOVE_RIGHT = 1
ACTION_MOVE_DOWN = 2
ACTION_MOVE_LEFT = 3
ACTION_BARRIER_MAKER = 4
ACTION_HELLIFY = 5

_DIR_VECTORS = {
    ACTION_MOVE_UP: (-1, 0),
    ACTION_MOVE_RIGHT: (0, 1),
    ACTION_MOVE_DOWN: (1, 0),
    ACTION_MOVE_LEFT: (0, -1),
}


class ShoverWorldEnv(gym.Env):
    """Shover-World environment (selection-based pushing of boxes)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 render_mode=None,
                 n_rows=10, n_cols=10,
                 max_timestep=400,
                 number_of_boxes=10,
                 number_of_barriers=2,
                 number_of_lavas=1,
                 initial_stamina=1000.0,
                 initial_force=4.0,
                 unit_force=1.0,
                 perf_sq_initial_age=50,
                 map_path=None,
                 seed=None):

        # Parameters
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.max_timestep = max_timestep
        self.number_of_boxes = number_of_boxes
        self.number_of_barriers = number_of_barriers
        self.number_of_lavas = number_of_lavas
        self.initial_stamina = float(initial_stamina)
        self.initial_force = float(initial_force)
        self.unit_force = float(unit_force)
        self.perf_sq_initial_age = int(perf_sq_initial_age)
        self.map_path = map_path
        self._seed = seed

        # State
        self.grid = np.zeros((self.n_rows, self.n_cols), dtype=np.int32)
        # agent_pos represents the **selection cursor** (row, col)
        self.agent_pos = (0, 0)
        self.stamina = float(self.initial_stamina)
        self.timestep = 0
        # Active perfect squares: list of {'n': n, 'pos': (r,c), 'age': age}
        self.active_squares: List[Dict] = []

        # Non-stationary bookkeeping for push stationary logic
        # store triples (r, c, dir_int) of boxes that moved in that direction this step
        self.non_stationary_this_step = set()
        self.non_stationary_last_step = set()

        # Counters tracked in info
        self.boxes_destroyed = 0
        self.last_action_valid = True
        self.last_chain_k = 0
        self.last_initial_force_charged = False
        self.last_lava_destroyed_this_step = 0

        # RNG
        self.np_random = np.random.default_rng(self._seed)

        # Spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=-100, high=100,
                               shape=(self.n_rows, self.n_cols), dtype=np.int32),
            "agent": spaces.Box(low=0, high=max(self.n_rows, self.n_cols),
                                shape=(2,), dtype=np.int32),
            "stamina": spaces.Box(low=0, high=np.finfo(np.float32).max,
                                  shape=(1,), dtype=np.float32)
        })

        # Rendering
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.cell_size = 50

    # ---------- Helpers ----------
    def _get_obs(self):
        return {
            "grid": self.grid.copy(),
            "agent": np.array([self.agent_pos[0], self.agent_pos[1]], dtype=np.int32),
            "stamina": np.array([self.stamina], dtype=np.float32)
        }

    def _get_info(self):
        return {
            "timestep": self.timestep,
            "stamina": self.stamina,
            "boxes_remaining": int(np.count_nonzero((self.grid > 0) & (self.grid <= 10))),
            "boxes_destroyed": int(self.boxes_destroyed),
            "last_action_valid": bool(self.last_action_valid),
            "chain_length_k": int(self.last_chain_k),
            "initial_force_charged": bool(self.last_initial_force_charged),
            "lava_destroyed_this_step": int(self.last_lava_destroyed_this_step),
            "perfect_squares_available": [(sq['n'], tuple(sq['pos'])) for sq in self.active_squares]
        }

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        used_seed = seed if seed is not None else self._seed
        self.np_random = np.random.default_rng(used_seed)

        # Load or generate map
        if self.map_path:
            self._load_map(self.map_path)
        else:
            self._generate_random_map()

        # Initialise dynamic state
        self.stamina = float(self.initial_stamina)
        self.timestep = 0
        self.non_stationary_this_step.clear()
        self.non_stationary_last_step.clear()
        self.boxes_destroyed = 0
        self.last_action_valid = True
        self.last_chain_k = 0
        self.last_initial_force_charged = False
        self.last_lava_destroyed_this_step = 0

        self._update_and_age_squares()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        # Increment step counter and rotate non-stationary bookkeeping
        self.timestep += 1
        self.non_stationary_last_step = self.non_stationary_this_step.copy()
        self.non_stationary_this_step.clear()

        # reset per-step fields
        self.last_action_valid = True
        self.last_chain_k = 0
        self.last_initial_force_charged = False
        self.last_lava_destroyed_this_step = 0

        reward = 0.0
        stamina_cost = 0.0
        stamina_gain = 0.0

        # Movement/push actions (0-3)
        if ACTION_MOVE_UP <= action <= ACTION_MOVE_LEFT:
            dr, dc = _DIR_VECTORS[action]
            sel_r, sel_c = self.agent_pos
            # If selected cell contains a box -> try to push chain starting from it
            if 1 <= int(self.grid[sel_r, sel_c]) <= 10:
                cost, refund, moved, k = self._push_chain((sel_r, sel_c), action)
                stamina_cost += cost
                stamina_gain += refund
                self.last_chain_k = k
                self.last_action_valid = moved
                # Determine if initial_force was charged (cost includes it)
                self.last_initial_force_charged = False
                if moved and k > 0:
                    if cost - (self.unit_force * k) > 1e-6:
                        self.last_initial_force_charged = True
            else:
                # Otherwise move the selection cursor relative to the grid
                new_r = sel_r + dr
                new_c = sel_c + dc
                if 0 <= new_r < self.n_rows and 0 <= new_c < self.n_cols:
                    self.agent_pos = (new_r, new_c)
                else:
                    self.last_action_valid = False

        elif action == ACTION_BARRIER_MAKER:
            if not self._is_action_available(ACTION_BARRIER_MAKER):
                self.last_action_valid = False
            else:
                # pick oldest square (active_squares sorted by age desc)
                if not self.active_squares:
                    self.last_action_valid = False
                else:
                    oldest = self.active_squares[0]
                    n = oldest['n']
                    r0, c0 = oldest['pos']
                    # Convert n x n boxes to barriers
                    for rr in range(r0, r0 + n):
                        for cc in range(c0, c0 + n):
                            self.grid[rr, cc] = CELL_BARRIER
                    # Award stamina
                    stamina_gain += (n * n)
                    # Recompute squares
                    self._update_and_age_squares()

        elif action == ACTION_HELLIFY:
            if not self._is_action_available(ACTION_HELLIFY):
                self.last_action_valid = False
            else:
                # find oldest square with n > 2
                found = None
                for sq in self.active_squares:
                    if sq['n'] > 2:
                        found = sq
                        break
                if found is None:
                    self.last_action_valid = False
                else:
                    n = found['n']
                    r0, c0 = found['pos']
                    destroyed = 0
                    # boundary -> empty, interior -> lava
                    for rr in range(r0, r0 + n):
                        for cc in range(c0, c0 + n):
                            if rr == r0 or rr == r0 + n - 1 or cc == c0 or cc == c0 + n - 1:
                                if 1 <= self.grid[rr, cc] <= 10:
                                    destroyed += 1
                                self.grid[rr, cc] = CELL_EMPTY
                            else:
                                if 1 <= self.grid[rr, cc] <= 10:
                                    destroyed += 1
                                self.grid[rr, cc] = CELL_LAVA
                    self.boxes_destroyed += destroyed
                    # Update squares
                    self._update_and_age_squares()

        # Baseline stamina cost for any action
        baseline_cost = 1.0
        self.stamina -= (baseline_cost + stamina_cost - stamina_gain)

        # Update squares, apply dissolution if needed
        self._update_and_age_squares()
        dissolved = self._apply_dissolution()
        if dissolved:
            self._update_and_age_squares()

        # Reward: give reward for boxes destroyed into lava this step
        if self.last_lava_destroyed_this_step > 0:
            reward += self.initial_force * self.last_lava_destroyed_this_step

        # Termination checks
        terminated = False
        if self.stamina <= 0:
            terminated = True
        if int(np.count_nonzero((self.grid > 0) & (self.grid <= 10))) == 0:
            terminated = True

        truncated = False
        if self.timestep >= self.max_timestep:
            truncated = True

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ---------- Map loading / generation ----------
    def _load_map(self, map_path: str):
        try:
            with open(map_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            grid_data = []
            agent_found = False
            agent_pos = (0, 0)
            for r, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith('//'):
                    continue
                tokens = line.split()
                try:
                    row = [int(val) for val in tokens]
                except ValueError:
                    row = []
                    for c, tok in enumerate(tokens):
                        if tok == '.':
                            row.append(CELL_EMPTY)
                        elif tok.upper() == 'B':
                            row.append(10)
                        elif tok == '#':
                            row.append(CELL_BARRIER)
                        elif tok.upper() == 'L':
                            row.append(CELL_LAVA)
                        elif tok.upper() == 'A':
                            row.append(CELL_EMPTY)
                            agent_found = True
                            agent_pos = (r, c)
                        else:
                            try:
                                row.append(int(tok))
                            except Exception:
                                row.append(CELL_EMPTY)
                grid_data.append(row)
            self.grid = np.array(grid_data, dtype=np.int32)
            self.n_rows, self.n_cols = self.grid.shape
            if agent_found:
                self.agent_pos = agent_pos
            else:
                empties = np.argwhere(self.grid == CELL_EMPTY)
                if empties.size > 0:
                    chosen = empties[int(self.np_random.integers(0, len(empties)))]
                    self.agent_pos = (int(chosen[0]), int(chosen[1]))
                else:
                    self.agent_pos = (0, 0)
        except Exception as e:
            # fallback
            print(f"Error loading map: {e}. Using random map.")
            self._generate_random_map()

    def _generate_random_map(self):
        self.grid = np.zeros((self.n_rows, self.n_cols), dtype=np.int32)
        # Place selection cursor at random cell
        r = int(self.np_random.integers(0, self.n_rows))
        c = int(self.np_random.integers(0, self.n_cols))
        self.agent_pos = (r, c)

        # helper: sample empty cells excluding selection
        empties = [(i, j) for i in range(self.n_rows) for j in range(self.n_cols) if (i, j) != self.agent_pos]
        self.np_random.shuffle(empties)

        # place boxes
        for i in range(min(self.number_of_boxes, len(empties))):
            rr, cc = empties.pop()
            self.grid[rr, cc] = 10

        # place barriers
        for i in range(min(self.number_of_barriers, len(empties))):
            rr, cc = empties.pop()
            self.grid[rr, cc] = CELL_BARRIER

        # place lavas
        for i in range(min(self.number_of_lavas, len(empties))):
            rr, cc = empties.pop()
            self.grid[rr, cc] = CELL_LAVA

    # ---------- Pushing mechanics ----------
    def _push_chain(self, start_box_pos: Tuple[int, int], direction: int):
        """
        Attempt to push the chain starting at start_box_pos in direction.
        Returns (push_cost, lava_refund, moved_bool, k)
        """
        dr, dc = _DIR_VECTORS[direction]
        chain = []
        r, c = start_box_pos
        # collect contiguous boxes in that direction
        while 0 <= r < self.n_rows and 0 <= c < self.n_cols and 1 <= int(self.grid[r, c]) <= 10:
            chain.append((r, c))
            r += dr
            c += dc
        beyond_r, beyond_c = r, c

        # OOB beyond chain => invalid push
        if not (0 <= beyond_r < self.n_rows and 0 <= beyond_c < self.n_cols):
            return (0.0, 0.0, False, 0)

        beyond_val = int(self.grid[beyond_r, beyond_c])
        # if beyond is barrier or box -> invalid
        if beyond_val == CELL_BARRIER or (1 <= beyond_val <= 10):
            return (0.0, 0.0, False, 0)

        k = len(chain)
        if k == 0:
            return (0.0, 0.0, False, 0)

        head = chain[0]  # box adjacent to the selection
        is_stationary = (head[0], head[1], direction) not in self.non_stationary_last_step

        push_cost = 0.0
        if is_stationary:
            push_cost += self.initial_force
        push_cost += self.unit_force * k

        lava_refund = 0.0
        lava_destroyed = 0

        # move boxes from tail to head
        for (br, bc) in reversed(chain):
            new_r = br + dr
            new_c = bc + dc
            # if destination is lava -> destroy the box
            if self.grid[new_r, new_c] == CELL_LAVA:
                lava_refund += self.initial_force
                lava_destroyed += 1
                self.grid[br, bc] = CELL_EMPTY
                self.boxes_destroyed += 1
            else:
                # move box to new cell
                self.grid[new_r, new_c] = self.grid[br, bc]
                # mark as non-stationary in this direction
                self.non_stationary_this_step.add((new_r, new_c, direction))
                self.grid[br, bc] = CELL_EMPTY

        # selection remains the same (you select the same position), but last action valid true
        self.last_lava_destroyed_this_step = lava_destroyed

        return (push_cost, lava_refund, True, k)

    # ---------- Perfect-square detection ----------
    def _find_perfect_squares(self):
        """Return list of {'n', 'pos'} for valid perfect squares."""
        is_box = ((self.grid > 0) & (self.grid <= 10))
        dp = np.zeros_like(self.grid, dtype=np.int32)
        found = []
        seen = set()
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if is_box[r, c]:
                    if r == 0 or c == 0:
                        dp[r, c] = 1
                    else:
                        dp[r, c] = min(dp[r - 1, c], dp[r, c - 1], dp[r - 1, c - 1]) + 1
                    maxsize = dp[r, c]
                    # consider sizes >= 2 only (n>=2 can be barrier-maker)
                    for size in range(maxsize, 1, -1):
                        top_r = r - size + 1
                        top_c = c - size + 1
                        key = (size, top_r, top_c)
                        if key in seen:
                            continue
                        if self._check_perimeter((top_r, top_c), size, is_box):
                            found.append({'n': size, 'pos': (top_r, top_c)})
                            seen.add(key)
                else:
                    dp[r, c] = 0
        return found

    def _check_perimeter(self, top_left_pos: Tuple[int, int], n: int, is_box_grid: np.ndarray):
        """Return True if the outside 8-neighborhood around the square contains no boxes."""
        r0, c0 = top_left_pos
        # row above
        row_above = r0 - 1
        if row_above >= 0:
            for cc in range(c0 - 1, c0 + n + 1):
                if 0 <= cc < self.n_cols and is_box_grid[row_above, cc]:
                    return False
        # row below
        row_below = r0 + n
        if row_below < self.n_rows:
            for cc in range(c0 - 1, c0 + n + 1):
                if 0 <= cc < self.n_cols and is_box_grid[row_below, cc]:
                    return False
        # left col
        col_left = c0 - 1
        if col_left >= 0:
            for rr in range(r0, r0 + n):
                if 0 <= rr < self.n_rows and is_box_grid[rr, col_left]:
                    return False
        # right col
        col_right = c0 + n
        if col_right < self.n_cols:
            for rr in range(r0, r0 + n):
                if 0 <= rr < self.n_rows and is_box_grid[rr, col_right]:
                    return False
        return True

    # ---------- Square aging & dissolution ----------
    def _update_and_age_squares(self):
        found_squares = self._find_perfect_squares()
        new_active = []
        old_lookup = {(sq['n'], tuple(sq['pos'])): sq.get('age', 0) for sq in self.active_squares}
        for sq in found_squares:
            key = (sq['n'], tuple(sq['pos']))
            if key in old_lookup:
                new_age = old_lookup[key] + 1
            else:
                new_age = 0
            new_active.append({'n': sq['n'], 'pos': tuple(sq['pos']), 'age': new_age})
        # sort by age descending (oldest first)
        self.active_squares = sorted(new_active, key=lambda x: x['age'], reverse=True)

    def _apply_dissolution(self):
        dissolved_one = False
        remaining = []
        for sq in self.active_squares:
            if sq['age'] >= self.perf_sq_initial_age:
                dissolved_one = True
                r0, c0 = sq['pos']
                n = sq['n']
                for rr in range(r0, r0 + n):
                    for cc in range(c0, c0 + n):
                        self.grid[rr, cc] = CELL_EMPTY
                # do not count these as boxes_destroyed per spec
            else:
                remaining.append(sq)
        self.active_squares = remaining
        return dissolved_one

    # ---------- Action availability ----------
    def _is_action_available(self, action):
        if action == ACTION_BARRIER_MAKER:
            return any(sq['n'] >= 2 for sq in self.active_squares)
        if action == ACTION_HELLIFY:
            return any(sq['n'] > 2 for sq in self.active_squares)
        return True

    # ---------- Rendering ----------
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Shover-World")
            self.window = pygame.display.set_mode(
                (self.n_cols * self.cell_size, self.n_rows * self.cell_size + 80)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(
            (self.n_cols * self.cell_size, self.n_rows * self.cell_size + 80)
        )
        canvas.fill((255, 255, 255))

        # Draw cells
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                val = int(self.grid[r, c])
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size,
                                   self.cell_size, self.cell_size)
                color = (240, 240, 240)  # empty
                if val == CELL_LAVA:
                    color = (255, 120, 0)
                elif val == CELL_BARRIER:
                    color = (80, 80, 80)
                elif 1 <= val <= 10:
                    color = (140, 85, 40)  # box brown
                pygame.draw.rect(canvas, color, rect)
                pygame.draw.rect(canvas, (0, 0, 0), rect, 1)

        # Highlight selected cell (no blue ball)
        sel_r, sel_c = self.agent_pos
        if 0 <= sel_r < self.n_rows and 0 <= sel_c < self.n_cols:
            sel_rect = pygame.Rect(sel_c * self.cell_size, sel_r * self.cell_size,
                                   self.cell_size, self.cell_size)
            pygame.draw.rect(canvas, (0, 200, 0), sel_rect, 3)  # green outline

        # HUD
        font = pygame.font.SysFont("Arial", 18)
        hud_y = self.n_rows * self.cell_size + 5
        info_texts = [
            f"Step: {self.timestep}",
            f"Stamina: {self.stamina:.1f}",
            f"Boxes remaining: {int(np.count_nonzero((self.grid > 0) & (self.grid <= 10)))}",
            f"Boxes destroyed: {self.boxes_destroyed}"
        ]
        for i, t in enumerate(info_texts):
            surf = font.render(t, True, (0, 0, 0))
            canvas.blit(surf, (5 + i * 220, hud_y))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window:
            pygame.display.quit()
            pygame.quit()
