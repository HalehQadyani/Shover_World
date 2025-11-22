# gui.py
"""
Polished GUI controller for Shover-World (Strict Spec Compliance)
- Sends actions z=1..6 directly (no 0-index mapping).
"""

import pygame
import numpy as np
from environment import ShoverWorldEnv

# Action Constants (1-6)
try:
    from environment import ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT, ACTION_BARRIER_MAKER, ACTION_HELLIFY
except ImportError:
    ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT = 1, 2, 3, 4
    ACTION_BARRIER_MAKER, ACTION_HELLIFY = 5, 6

# Map keys to Action IDs (1..6)
_KEY_TO_Z = {
    pygame.K_UP: ACTION_UP, pygame.K_w: ACTION_UP,
    pygame.K_RIGHT: ACTION_RIGHT, pygame.K_d: ACTION_RIGHT,
    pygame.K_DOWN: ACTION_DOWN, pygame.K_s: ACTION_DOWN,
    pygame.K_LEFT: ACTION_LEFT, pygame.K_a: ACTION_LEFT,
    pygame.K_b: ACTION_BARRIER_MAKER,
    pygame.K_h: ACTION_HELLIFY,
}


class ShoverWorldGUI:
    def __init__(self, map_path=None, render_mode='human'):
        pygame.init()
        pygame.display.set_caption("Shover-World (Milestone 1)")

        # Initialize Environment
        self.env = ShoverWorldEnv(
            render_mode=render_mode,
            map_path=map_path,
            initial_stamina=1000.0,
            initial_force=40.0,  # Spec default is 40, not 4
            unit_force=10.0  # Spec default is 10, not 1
        )

        self.obs, self.info = self.env.reset()

        # Selection Cursor Logic
        try:
            self.selected = tuple(self.env.agent_pos)
        except AttributeError:
            self.selected = (0, 0)

        self.running = True
        self.clock = pygame.time.Clock()

        print("--- Controls ---")
        print("Arrows/WASD: Move/Push | B: Barrier | H: Hellify")
        print("Click: Select Cell | R: Reset | Q: Quit")

    def run(self):
        while self.running:
            action_to_send = None

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        self.running = False
                    elif event.key == pygame.K_r:
                        self.obs, self.info = self.env.reset()
                        self.selected = tuple(self.env.agent_pos)
                        print("--- Reset ---")
                    else:
                        # Get z value (1..6)
                        z_val = _KEY_TO_Z.get(event.key)
                        if z_val is not None:
                            # SEND DIRECTLY (Do not subtract 1)
                            action_to_send = (self.selected, z_val)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    # Assume 40px cell size from environment defaults
                    r, c = my // 40, mx // 40
                    if 0 <= r < self.env.n_rows and 0 <= c < self.env.n_cols:
                        self.selected = (r, c)
                        # Visually sync agent pos to selection for rendering highlight
                        self.env.agent_pos = (r, c)

            if action_to_send:
                # Step the environment
                obs, reward, term, trunc, info = self.env.step(action_to_send)
                self.obs, self.info = obs, info

                # Sync selection if the agent moved automatically
                self.selected = tuple(self.env.agent_pos)

                # Debug output
                valid = info.get('last_action_valid', True)
                stamina = info.get('stamina', 0)
                print(
                    f"Step: {info.get('timestep')} | Action: {action_to_send[1]} | Valid: {valid} | Stamina: {stamina:.1f}")

                if term or trunc:
                    print("Episode Finished. Resetting...")
                    self.obs, self.info = self.env.reset()
                    self.selected = tuple(self.env.agent_pos)

            self.env.render()
            self.clock.tick(30)

        self.env.close()
        pygame.quit()


if __name__ == "__main__":
    # You can test with: python gui.py
    gui = ShoverWorldGUI(map_path=None)
    gui.run()
