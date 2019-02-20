"""The Bridge Building environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from gym_pycolab import pycolab_env

from gym_tool_use import games
from gym_tool_use import utils


BRIDGE_BUILDING_TEMPLATE = [
    '#########',
    '#       #', 
    '#       #', 
    '#       #', 
    '#WWWWWWW#',
    '#       #', 
    '#       #', 
    '#       #', 
    '#########'
]
WATER_BOX_POSITIONS = [
    (x, y) 
    for y in range(1, len(BRIDGE_BUILDING_TEMPLATE) - 1)
    for x in range(2, 4)]


def generate_bridge_building_art(num_boxes, np_random=np.random):
    """Generate bridge building art."""

    assert num_boxes < 10, '`num_boxes` must be less than 10.'
    assert (num_boxes % 2) != 0, '`num_boxes` must be odd.'

    x_positions = [1, len(BRIDGE_BUILDING_TEMPLATE) - 2]

    player_side, goal_side = 0, 1

    # Generate player position
    player_y_position = np_random.randint(1, len(BRIDGE_BUILDING_TEMPLATE) - 1)
    player_position = (x_positions[player_side], player_y_position)

    # Generate goal position
    goal_y_position = np_random.randint(1, len(BRIDGE_BUILDING_TEMPLATE) - 1)
    goal_position = (x_positions[goal_side], goal_y_position)

    # Generate box positions
    box_side_positions = list(WATER_BOX_POSITIONS)
    box_position_indices = np_random.choice(
        len(box_side_positions), size=num_boxes, replace=False)
    box_positions = [box_side_positions[box_position_index] 
                    for box_position_index in box_position_indices]

    # Assign random numbers to the boxes.
    box_ids = np_random.choice(len(utils.WATER_BOXES), size=num_boxes, replace=False)

    # Paint positions
    art = list(BRIDGE_BUILDING_TEMPLATE)
    art = [list(row) for row in art]
    art = utils.paint(art, [player_position], ['P'])
    art = utils.paint(art, [goal_position], ['G'])
    art = utils.paint(art, box_positions, [str(box_id) for box_id in box_ids])
    art = [''.join(row) for row in art]
    return art


class BridgeBuildingEnv(pycolab_env.PyColabEnv):
    """Bridge building game."""

    def __init__(self, max_iterations=20):
        self.np_random = None
        super(BridgeBuildingEnv, self).__init__(
            game_factory=lambda: games.make_tool_use_game(
                generate_bridge_building_art(
                    1, self.np_random if self.np_random else np.random)),
            max_iterations=max_iterations, 
            default_reward=0.,
            action_space=utils.ACTION_SPACE,
            resize_scale=32,
            delay=200,
            colors=utils.COLORS)


if __name__ == "__main__":
    env = BridgeBuildingEnv()
    env.reset()
    env.render()
    env.close()