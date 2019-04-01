"""The Tool Use Environment.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

from gym_pycolab import pycolab_env

from gym_tool_use import games
from gym_tool_use import bridge_building_toolset
from gym_tool_use import utils


TOOL_USE_TEMPLATE = [
    '        ',
    '        ', 
    '        ', 
    '        ', 
    '        ',
    '        ', 
    '        ', 
    '        ',
]

ALL_POSITIONS = list(itertools.product(list(
    range(0, len(TOOL_USE_TEMPLATE))), repeat=2))


def generate_tool_use_art(toolsets, 
                          np_random=np.random):
    """Generate tool-use art given the toolsets.
    
    Args:
        toolsets: list of `ToolSet` that render tool interactions.
        np_random: random source.
    
    Returns:
        game art.
    """
    prefill_positions = {}

    # Shuffle positions
    all_positions = list(ALL_POSITIONS)
    np_random.shuffle(all_positions)

    # 2d grid.
    art = list(TOOL_USE_TEMPLATE)
    art = [list(row) for row in art]
    what_lies_beneath = [[' '] * len(row) for row in art]

    # Paint the goal.
    goal_position = (6, np_random.choice(len(art)))
    all_positions.remove(goal_position)
    art = utils.paint(art, [goal_position], ['G'])

    # Paint the player.
    player_position = (1, np_random.choice(len(art)))
    all_positions.remove(player_position)
    art = utils.paint(art, [player_position], ['P'])

    # Generate one successful sequence of actions.
    path = utils.bfs(
        art, 
        player_position, 
        len(TOOL_USE_TEMPLATE), 
        len(TOOL_USE_TEMPLATE))
    path.pop(0)
    path.pop(-1)

    # Render toolsets.
    for toolset in toolsets:
        art, what_lies_beneath, prefill_positions, all_positions = toolset.paint(
            art, 
            what_lies_beneath,
            player_position,
            goal_position,
            all_positions,
            prefill_positions,
            shortest_path=list(path),
            np_random=np_random)

    art = [''.join(row) for row in art]
    what_lies_beneath = [''.join(row) for row in what_lies_beneath]
    return art, what_lies_beneath, prefill_positions


class ToolUseEnv(pycolab_env.PyColabEnv):
    """Tool-use 'practice' game."""

    def __init__(self, 
                 toolsets=[bridge_building_toolset.BridgeBuildingToolSet()], 
                 observation_type='layers',
                 max_iterations=20):
        toolsets = set(toolsets)
        merge_layer_groups = [set([str(box) for box in range(len(utils.WATER_BOXES))])]
        self.np_random = None
        super(ToolUseEnv, self).__init__(
            game_factory=lambda: games.make_tool_use_game(
                *generate_tool_use_art(
                    toolsets, 
                    self.np_random if self.np_random else np.random)),
            max_iterations=max_iterations, 
            default_reward=0.,
            action_space=utils.ACTION_SPACE,
            resize_scale=32,
            observation_type=observation_type,
            delay=200,
            colors=utils.COLORS,
            exclude_from_state=set([' ']),
            merge_layer_groups=merge_layer_groups)


if __name__ == "__main__":
    w,d,a,s = 0,3,2,1

    np.random.seed(42)
    env = ToolUseEnv(
        toolsets=[bridge_building_toolset.BridgeBuildingToolSet()])
    for _ in range(10):
        state = env.reset()
        env.render()
        print(env._observation_order)
        print(state.shape)
        print(state[:, :, 0])
        for action in [w, d, d, d, w, w, a, a, a, s, s]:
            state, _, _, _ = env.step(action)
            print(state[:, :, 0])
            env.render()
    env.close()