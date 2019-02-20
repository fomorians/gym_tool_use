"""Base tool-use environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

from gym_pycolab import pycolab_env

from gym_tool_use import games
from gym_tool_use import toolsets
from gym_tool_use import utils


TOOL_USE_TEMPLATE = [
    '#########',
    '#       #', 
    '#       #', 
    '#       #', 
    '#       #',
    '#       #', 
    '#       #', 
    '#       #', 
    '#########'
]

ALL_POSITIONS = list(itertools.product(list(
    range(1, len(TOOL_USE_TEMPLATE) - 1)), repeat=2))


def generate_tool_use_art(toolsets, np_random=np.random):
    """Generate tool-use art given the toolsets."""
    # Shuffle positions
    all_positions = list(ALL_POSITIONS)
    np_random.shuffle(all_positions)

    art = list(TOOL_USE_TEMPLATE)
    art = [list(row) for row in art]

    # Paint the goal.
    goal_position = all_positions.pop()
    art = utils.paint(art, [goal_position], ['G'])

    # Paint the player.
    player_position = all_positions.pop()
    art = utils.paint(art, [player_position], ['P'])

    # Generate one successful sequence of actions.
    path = utils.bfs(
        art, player_position, len(TOOL_USE_TEMPLATE), len(TOOL_USE_TEMPLATE))
    path.pop(0)
    path.pop(-1)

    for coordinate in path:
        all_positions.remove(coordinate)

    for toolset in toolsets:
        art, all_positions = toolset.paint(
            art, 
            player_position,
            goal_position,
            all_positions,
            np_random=np_random)

    art = [''.join(row) for row in art]
    return art


class ToolUseEnv(pycolab_env.PyColabEnv):
    """Tool-use 'practice' game."""

    def __init__(self, toolsets=[toolsets.BridgeBuildingToolSet(3)], max_iterations=20):
        toolsets = set(toolsets)
        self.np_random = None
        super(ToolUseEnv, self).__init__(
            game_factory=lambda: games.make_tool_use_game(
                generate_tool_use_art(
                    toolsets, 
                    self.np_random if self.np_random else np.random)),
            max_iterations=max_iterations, 
            default_reward=0.,
            action_space=utils.ACTION_SPACE,
            resize_scale=32,
            delay=200,
            colors=utils.COLORS)


if __name__ == "__main__":
    w,d,a,s = 0,3,2,1

    np.random.seed(42)
    env = ToolUseEnv(
        toolsets=[toolsets.BridgeBuildingToolSet(4)])
    env.reset()
    env.render()
    for action in [s, s, a, w, d, w, a, a]:
        _, reward, _, info = env.step(action)
        print(reward)
        env.render()
    env.close()