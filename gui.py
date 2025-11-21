# gui.py
"""
GUI controller for Shover-World (selection-based control)

Features:
 - Mouse click selects ANY cell (box or empty).
 - Arrow keys / WASD: apply movement/push relative to selected cell.
 - B / H: special abilities.
 - R: reset episode.
 - Q: quit.
"""

import pygame
import numpy as np
from environment import (
    ShoverWorldEnv,
    ACTION_MOVE_UP,
    ACTION_MOVE_RIGHT,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_BARRIER_MAKER,
    ACTION_HELLIFY
)


class ShoverWorldGUI:
    def __init__(self, map_path=None):
        pygame.init()

        self.env = ShoverWorldEnv(
            render_mode='human',
            map_path=map_path,
            initial_stamina=1000,
            initial_force=4.0,
            unit_force=1.0,
            max_timestep=400
        )

        self.obs, self.info = self.env.reset()
        self.running = True

        print("--- Shover-World Manual Control ---")
        print("Controls:")
        print("  WASD / Arrows : Push/Move selection-based")
        print("  B             : Barrier Maker")
        print("  H             : Hellify")
        print("  Mouse Click   : SELECT cell (not teleport)")
        print("  R             : Reset Episode")
        print("  Q             : Quit")
        print("-----------------------------------")

    # ----------------------------------
    #  Keyboard → Action mapping
    # ----------------------------------
    def get_action_from_keyboard(self, key):
        if key in (pygame.K_UP, pygame.K_w):
            return ACTION_MOVE_UP
        if key in (pygame.K_RIGHT, pygame.K_d):
            return ACTION_MOVE_RIGHT
        if key in (pygame.K_DOWN, pygame.K_s):
            return ACTION_MOVE_DOWN
        if key in (pygame.K_LEFT, pygame.K_a):
            return ACTION_MOVE_LEFT
        if key == pygame.K_b:
            return ACTION_BARRIER_MAKER
        if key == pygame.K_h:
            return ACTION_HELLIFY
        return None

    # ----------------------------------
    #  Mouse click selects target cell
    # ----------------------------------
    def handle_mouse_click(self, pos):
        mx, my = pos
        col = mx // self.env.cell_size
        row = my // self.env.cell_size

        # Ensure click is within grid (not HUD)
        if 0 <= row < self.env.n_rows and 0 <= col < self.env.n_cols:
            self.env.agent_pos = (row, col)
            print(f"Selected cell -> ({row}, {col})")

    # ----------------------------------
    #  Main GUI loop
    # ----------------------------------
    def run(self):
        while self.running:

            action = None

            # ---------- Event Handling ----------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_click(pygame.mouse.get_pos())

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.running = False

                    elif event.key == pygame.K_r:
                        self.obs, self.info = self.env.reset()
                        print("Episode Reset.")

                    action = self.get_action_from_keyboard(event.key)

            # ---------- Step the environment ----------
            if action is not None:
                self.obs, reward, terminated, truncated, info = self.env.step(action)

                print(f"Step {info['timestep']} | Action: {action} | Reward: {reward:.2f} | Stamina: {info['stamina']}")
                if not info["last_action_valid"]:
                    print("  -> Invalid Action!")
                if info["lava_destroyed_this_step"] > 0:
                    print(f"  -> Burned {info['lava_destroyed_this_step']} box(es)!")

                if terminated or truncated:
                    print("Episode ended — resetting.")
                    self.obs, self.info = self.env.reset()

            # ---------- Render ----------
            self.env.render()

        # Cleanup
        self.env.close()
        pygame.quit()

