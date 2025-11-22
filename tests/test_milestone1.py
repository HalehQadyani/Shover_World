import pytest
import numpy as np
from environment import ShoverWorldEnv, ACTION_RIGHT, CELL_BOX, CELL_EMPTY, CELL_LAVA


def test_push_cost_stationary():
    """Milestone 1: Verify Initial Force charged for stationary push."""
    env = ShoverWorldEnv(initial_force=40, unit_force=10)

    # FIX: Must call reset() to initialize the grid array before modifying it
    env.reset()

    # Now we can overwrite the map for our specific test case
    env.grid.fill(CELL_EMPTY)
    env.grid[0, 1] = CELL_BOX
    env.agent_pos = (0, 0)

    # Stationary push (first move)
    # Action format: ((target_r, target_c), action_id)
    obs, r, term, trunc, info = env.step(((0, 0), ACTION_RIGHT))

    # Cost calculation:
    # Baseline move cost (0) + Push Cost
    # Push Cost = Initial Force (40) + Unit Force (10) * 1 box = 50
    # Expected Stamina = 1000 - 50 = 950
    assert info['stamina'] == 950.0
    assert info['initial_force_charged'] == True


def test_push_cost_non_stationary():
    """Milestone 1: Verify Initial Force WAIVED for subsequent push."""
    env = ShoverWorldEnv(initial_force=40, unit_force=10)
    env.reset()  # FIX: Initialize grid

    env.grid.fill(CELL_EMPTY)
    env.grid[0, 1] = CELL_BOX
    env.agent_pos = (0, 0)

    # 1. First Push (Stationary)
    # Box moves from (0,1) to (0,2)
    env.step(((0, 0), ACTION_RIGHT))

    # 2. Second Push (Non-Stationary)
    # Agent is now at (0,1). Box is at (0,2).
    # We push the box from (0,2) to (0,3).
    obs, r, term, trunc, info = env.step(((0, 1), ACTION_RIGHT))

    # Cost calculation for 2nd push:
    # Initial Force (WAIVED because it moved last step) + Unit Force (10) * 1 = 10
    # Total lost = 50 (1st push) + 10 (2nd push) = 60
    # Expected Stamina = 1000 - 60 = 940
    assert info['stamina'] == 940.0
    assert info['initial_force_charged'] == False


def test_lava_refund():
    """Milestone 1: Verify lava refund logic."""
    env = ShoverWorldEnv(initial_force=40, unit_force=10)
    env.reset()  # FIX: Initialize grid

    env.grid.fill(CELL_EMPTY)
    env.grid[0, 1] = CELL_BOX
    env.grid[0, 2] = CELL_LAVA
    env.agent_pos = (0, 0)

    # Push box into lava
    obs, r, term, trunc, info = env.step(((0, 0), ACTION_RIGHT))

    # Cost Calculation:
    # Push Cost = 40 (Init) + 10 (Unit) = 50 cost.
    # Refund = 40 (Initial Force).
    # Net Change = -50 + 40 = -10.
    # Expected Stamina = 1000 - 10 = 990.
    assert info['stamina'] == 990.0
    assert info['lava_destroyed_this_step'] == 1