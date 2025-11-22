import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

# Constants
CELL_LAVA = -100
CELL_EMPTY = 0
CELL_BOX = 10
CELL_BARRIER = 100

ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3
ACTION_LEFT = 4
ACTION_BARRIER_MAKER = 5
ACTION_HELLIFY = 6


class ShoverWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, n_rows=6, n_cols=9,
                 max_timestep=400, initial_stamina=1000.0,
                 initial_force=40.0, unit_force=10.0,
                 number_of_boxes=10, number_of_barriers=5, number_of_lavas=3,
                 perf_sq_initial_age=10, map_path=None, seed=None):
        super().__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.max_timestep = max_timestep
        self.render_mode = render_mode

        # Mechanics Config
        self.initial_stamina = float(initial_stamina)
        self.initial_force = float(initial_force)
        self.unit_force = float(unit_force)
        self.perf_sq_initial_age = perf_sq_initial_age

        # Generation Config
        self.number_of_boxes = number_of_boxes
        self.number_of_barriers = number_of_barriers
        self.number_of_lavas = number_of_lavas

        self.map_path = map_path
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Action Space: Tuple ((r, c), z) where z is 1..6.
        self.action_space = spaces.Tuple((
            spaces.Box(low=0, high=max(n_rows, n_cols), shape=(2,), dtype=np.int32),
            spaces.Discrete(7)
        ))

        # Observation Space
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=-100, high=100, shape=(n_rows, n_cols), dtype=np.int32),
            "agent": spaces.Box(low=0, high=max(n_rows, n_cols), shape=(2,), dtype=np.int32),
            "stamina": spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype=np.float32),
            "previous_selected_position": spaces.Box(low=0, high=max(n_rows, n_cols), shape=(2,), dtype=np.int32),
            "previous_action": spaces.Box(low=0, high=6, shape=(1,), dtype=np.int32)
        })

        self.window = None
        self.clock = None
        self.grid = None
        self.agent_pos = (0, 0)
        self.current_stamina = 0.0
        self.timestep = 0

        # Momentum tracking: (r,c) -> {direction: bool}
        self.box_momentum = {}
        # Perfect squares: List of dicts
        self.perfect_squares = []

        # Info tracking
        self.last_action_valid = False
        self.chain_length = 0
        self.initial_force_charged = False
        self.lava_destroyed = 0
        self.prev_sel = (0, 0)
        self.prev_act = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

        self.timestep = 0
        self.current_stamina = self.initial_stamina
        self.box_momentum = {}
        self.last_action_valid = False

        if self.map_path and os.path.exists(self.map_path):
            self._load_map_from_file(self.map_path)
        else:
            self._generate_random_map()

        self.perfect_squares = self._detect_perfect_squares()
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.timestep += 1
        target_pos, act_type = action

        # Update previous action tracking for observation
        self.prev_sel = target_pos
        self.prev_act = act_type

        reward = 0.0
        terminated = False
        truncated = False

        # Reset info trackers
        self.lava_destroyed = 0
        self.chain_length = 0
        self.initial_force_charged = False
        self.last_action_valid = False
        stamina_cost = 0.0

        # 1. Capture momentum from previous step for COST calculation
        momentum_snapshot = self.box_momentum

        # 2. Prepare momentum accumulator for THIS step
        self.box_momentum = {}

        # --- MOVEMENT (1-4) ---
        if act_type in [ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT]:
            dr, dc = self._get_direction_vector(act_type)
            next_r, next_c = self.agent_pos[0] + dr, self.agent_pos[1] + dc

            if not self._in_bounds(next_r, next_c):
                self.last_action_valid = False
            elif self.grid[next_r, next_c] == CELL_BARRIER:
                self.last_action_valid = False
            elif self.grid[next_r, next_c] == CELL_EMPTY:
                self.agent_pos = (next_r, next_c)
                self.last_action_valid = True
                # FIXED: variable name typo corrected below
                self.current_stamina -= 0.0
            elif self.grid[next_r, next_c] == CELL_BOX:
                chain, landing = self._get_push_chain(next_r, next_c, dr, dc)
                if landing:
                    self.last_action_valid = True
                    self.chain_length = len(chain)

                    # Cost Calculation
                    head_box = chain[0]
                    is_stationary = True
                    if head_box in momentum_snapshot:
                        if momentum_snapshot[head_box].get(act_type, False):
                            is_stationary = False

                    cost = (self.unit_force * self.chain_length)
                    if is_stationary:
                        cost += self.initial_force
                        self.initial_force_charged = True
                    stamina_cost = cost

                    # Execute Move
                    self._move_chain(chain, dr, dc, act_type)
                    self.agent_pos = (next_r, next_c)

                    # Lava Check
                    land_r, land_c = landing
                    if self._in_bounds(land_r, land_c) and self.grid[land_r, land_c] == CELL_LAVA:
                        self.current_stamina += self.initial_force  # Refund
                        self.lava_destroyed += 1
                else:
                    self.last_action_valid = False

        # --- SPECIALS ---
        elif act_type == ACTION_BARRIER_MAKER:
            squares = self._detect_perfect_squares(min_n=2)
            if squares:
                sq = squares[0]  # Oldest
                n, (r, c) = sq['size'], sq['top_left']
                self.grid[r:r + n, c:c + n] = CELL_BARRIER
                self.current_stamina += (n ** 2)
                self.last_action_valid = True

        elif act_type == ACTION_HELLIFY:
            squares = self._detect_perfect_squares(min_n=3)
            if squares:
                sq = squares[0]
                n, (r, c) = sq['size'], sq['top_left']
                # Border empty, inner lava
                self.grid[r:r + n, c:c + n] = CELL_LAVA
                self.grid[r, c:c + n] = CELL_EMPTY
                self.grid[r + n - 1, c:c + n] = CELL_EMPTY
                self.grid[r:r + n, c] = CELL_EMPTY
                self.grid[r:r + n, c + n - 1] = CELL_EMPTY

                self.last_action_valid = True

        # Update State
        self.current_stamina -= stamina_cost
        self._manage_square_aging()

        # Termination
        box_count = np.count_nonzero(self.grid == CELL_BOX)
        if self.current_stamina <= 0:
            terminated = True
        if box_count == 0:
            terminated = True
        if self.timestep >= self.max_timestep:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        return {
            "grid": self.grid.copy(),
            "agent": np.array(self.agent_pos, dtype=np.int32),
            "stamina": np.array([self.current_stamina], dtype=np.float32),
            "previous_selected_position": np.array(self.prev_sel, dtype=np.int32),
            "previous_action": np.array([self.prev_act], dtype=np.int32)
        }

    def _get_info(self):
        return {
            "timestep": self.timestep,
            "stamina": self.current_stamina,
            "number_of_boxes": np.count_nonzero(self.grid == CELL_BOX),
            "last_action_valid": self.last_action_valid,
            "chain_length": self.chain_length,
            "initial_force_charged": self.initial_force_charged,
            "lava_destroyed_this_step": self.lava_destroyed,
            "perfect_squares_available": [(s['size'], s['top_left']) for s in self.perfect_squares]
        }

    def _in_bounds(self, r, c):
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def _get_direction_vector(self, action):
        if action == ACTION_UP: return -1, 0
        if action == ACTION_RIGHT: return 0, 1
        if action == ACTION_DOWN: return 1, 0
        if action == ACTION_LEFT: return 0, -1
        return 0, 0

    def _get_push_chain(self, start_r, start_c, dr, dc):
        chain = []
        curr_r, curr_c = start_r, start_c
        while self._in_bounds(curr_r, curr_c) and self.grid[curr_r, curr_c] == CELL_BOX:
            chain.append((curr_r, curr_c))
            curr_r += dr
            curr_c += dc

        if self._in_bounds(curr_r, curr_c):
            cell = self.grid[curr_r, curr_c]
            if cell == CELL_EMPTY or cell == CELL_LAVA:
                return chain, (curr_r, curr_c)
        return chain, None

    def _move_chain(self, chain, dr, dc, direction):
        # Move from last to first
        for r, c in reversed(chain):
            nr, nc = r + dr, c + dc
            if self.grid[nr, nc] != CELL_LAVA:
                self.grid[nr, nc] = CELL_BOX

            self.grid[r, c] = CELL_EMPTY

            # Mark momentum
            if (nr, nc) not in self.box_momentum:
                self.box_momentum[(nr, nc)] = {}
            self.box_momentum[(nr, nc)][direction] = True

    # --- Perfect Squares ---
    def _detect_perfect_squares(self, min_n=2):
        detected = []
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.grid[r, c] == CELL_BOX:
                    max_n = min(self.n_rows - r, self.n_cols - c)
                    for n in range(min_n, max_n + 1):
                        if self._is_perfect(r, c, n):
                            age = 0
                            for s in self.perfect_squares:
                                if s['top_left'] == (r, c) and s['size'] == n:
                                    age = s['age']
                            detected.append({'top_left': (r, c), 'size': n, 'age': age})
        return detected

    def _is_perfect(self, r, c, n):
        if not np.all(self.grid[r:r + n, c:c + n] == CELL_BOX): return False
        r_s, r_e = max(0, r - 1), min(self.n_rows, r + n + 1)
        c_s, c_e = max(0, c - 1), min(self.n_cols, c + n + 1)
        for i in range(r_s, r_e):
            for j in range(c_s, c_e):
                if r <= i < r + n and c <= j < c + n: continue  # Skip inner
                if self.grid[i, j] == CELL_BOX: return False
        return True

    def _manage_square_aging(self):
        current_squares = self._detect_perfect_squares(min_n=2)
        new_list = []
        for sq in current_squares:
            r, c = sq['top_left']
            n = sq['size']
            square_moved = False
            for i in range(n):
                for j in range(n):
                    if (r + i, c + j) in self.box_momentum:
                        square_moved = True

            if square_moved:
                sq['age'] = 0
            else:
                sq['age'] += 1

            if sq['age'] >= self.perf_sq_initial_age:
                self.grid[r:r + n, c:c + n] = CELL_EMPTY
            else:
                new_list.append(sq)
        self.perfect_squares = new_list

    def _generate_random_map(self):
        self.grid = np.zeros((self.n_rows, self.n_cols), dtype=np.int32)
        positions = [(r, c) for r in range(self.n_rows) for c in range(self.n_cols)]
        self._rng.shuffle(positions)

        self.agent_pos = positions.pop()

        for _ in range(min(len(positions), self.number_of_boxes)):
            r, c = positions.pop()
            self.grid[r, c] = CELL_BOX

        for _ in range(min(len(positions), self.number_of_barriers)):
            r, c = positions.pop()
            self.grid[r, c] = CELL_BARRIER

        for _ in range(min(len(positions), self.number_of_lavas)):
            r, c = positions.pop()
            self.grid[r, c] = CELL_LAVA

    def _load_map_from_file(self, path):
        with open(path, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("//")]

        try:
            [int(x) for x in lines[0].split()]
            is_sym = False
        except:
            is_sym = True

        grid_data = []
        for r, line in enumerate(lines):
            row = []
            if is_sym:
                for char in line:
                    if char == 'B':
                        row.append(CELL_BOX)
                    elif char == '#':
                        row.append(CELL_BARRIER)
                    elif char == 'L':
                        row.append(CELL_LAVA)
                    elif char == 'A':
                        self.agent_pos = (r, len(row))
                        row.append(CELL_EMPTY)
                    else:
                        row.append(CELL_EMPTY)
            else:
                tokens = line.split()
                for c, t in enumerate(tokens):
                    val = int(t)
                    row.append(val)
            grid_data.append(row)
        self.grid = np.array(grid_data, dtype=np.int32)
        self.n_rows, self.n_cols = self.grid.shape

    def render(self):
        if self.render_mode == "human":
            if self.window is None:
                import pygame
                pygame.init()
                pygame.display.set_caption("Shover-World")
                self.window = pygame.display.set_mode((self.n_cols * 40, self.n_rows * 40 + 80))
                self.clock = pygame.time.Clock()
            self._render_frame()

    def _render_frame(self):
        import pygame
        canvas = pygame.Surface((self.n_cols * 40, self.n_rows * 40 + 80))
        canvas.fill((30, 30, 30))

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                rect = pygame.Rect(c * 40, 50 + r * 40, 40, 40)
                val = self.grid[r, c]
                color = (50, 50, 50)
                if val == CELL_BOX:
                    color = (139, 69, 19)
                elif val == CELL_BARRIER:
                    color = (100, 100, 100)
                elif val == CELL_LAVA:
                    color = (255, 69, 0)

                pygame.draw.rect(canvas, color, rect)
                pygame.draw.rect(canvas, (200, 200, 200), rect, 1)

        ar, ac = self.agent_pos
        pygame.draw.circle(canvas, (0, 255, 0), (ac * 40 + 20, 50 + ar * 40 + 20), 12)

        font = pygame.font.SysFont('Arial', 18)
        hud = f"Stamina: {self.current_stamina} | Steps: {self.timestep}"
        canvas.blit(font.render(hud, True, (255, 255, 255)), (10, 10))

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window:
            import pygame
            pygame.display.quit()
            pygame.quit()