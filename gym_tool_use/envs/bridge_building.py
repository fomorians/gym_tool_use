"""The Bridge Building environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from gym_pycolab import pycolab_env

from gym_tool_use import games
from gym_tool_use import utils


BRIDGE_BUILDING_TEMPLATE = [
    '        ',
    '        ', 
    '        ', 
    '        ', 
    'WWWWWWWW',  # TODO(wenkesj): draw different water formations.
    '        ', 
    '        ', 
    '        ',
]


def generate_bridge_building_art(num_boxes, np_random=np.random):
    """Generate bridge building art."""

    assert num_boxes < 4, '`num_boxes` must be less than 4.'
    assert (num_boxes % 2) != 0, '`num_boxes` must be odd.'

    x_positions = [1, len(BRIDGE_BUILDING_TEMPLATE) - 2]
    player_side, goal_side = 0, 1
    c_l, c_r = 2, len(BRIDGE_BUILDING_TEMPLATE) - 2

    # Generate player position
    player_y_position = np_random.randint(c_l, c_r)
    player_position = (x_positions[player_side], player_y_position)

    # Generate goal position
    goal_y_position = np_random.randint(c_l, c_r)
    goal_position = (x_positions[goal_side], goal_y_position)

    # Generate box positions out of the shortest path.
    c_l_range = list(range(c_l - 1, min(player_y_position, goal_y_position)))
    c_r_range = list(range(max(player_y_position, goal_y_position) + 1, c_r + 1))
    box_side_positions = [
        (x, y)
        for x in list(range(2, 4))
        for y in c_l_range + c_r_range]
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

    def __init__(self, observation_type='layers', max_iterations=20):
        merge_layer_groups = [set([str(box) for box in range(len(utils.WATER_BOXES))])]
        self.np_random = None
        super(BridgeBuildingEnv, self).__init__(
            game_factory=lambda: games.make_tool_use_game(
                generate_bridge_building_art(
                    1, self.np_random if self.np_random else np.random)),
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
    np.random.seed(42)
    env = BridgeBuildingEnv(observation_type='layers')
    print(env._observation_order)
    env.seed(42)
    for _ in range(10):
        state = env.reset()
        print(env._observation_order)
        print(state.shape)
        print(state[:, :, 0])
        for action in [1, 2, 2, 2, 2, 2, 2]:
            env.render()
            state, _, _, _ = env.step(action)
            print(state[:, :, 0])
        env.render()
    env.close()