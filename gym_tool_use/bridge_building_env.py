"""The Bridge Building Environment.

In this environment, water and boxes can exist on the edge of the 
game.

The agent always starts at the top (row 1), and the goal is on the 
bottom (row last-1).

A box will spawn in +1 or +2 rows from the agent, but can not spawn 
in the columns between the agent and the goal.

A natural 'river' (water in all columns) exists +3 rows 
from the agent.
"""

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
    'WWWWWWWW',
    '        ', 
    '        ', 
    '        ',
]


def generate_bridge_building_art(num_boxes, np_random=np.random):
    """Generate bridge building art."""
    x_positions = [1, len(BRIDGE_BUILDING_TEMPLATE) - 2]
    player_side, goal_side = 0, 1
    c_l, c_r = 2, len(BRIDGE_BUILDING_TEMPLATE) - 2

    # Generate player position
    player_y_position = np_random.randint(c_l, c_r)
    player_x_position = x_positions[player_side]
    player_position = (player_x_position, player_y_position)

    # Generate goal position
    goal_y_position = np_random.randint(c_l, c_r)
    goal_x_position = x_positions[goal_side]
    goal_position = (goal_x_position, goal_y_position)

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
    box_ids = np_random.choice(len(utils.BOXES), size=num_boxes, replace=False)

    # Paint positions
    art = list(BRIDGE_BUILDING_TEMPLATE)
    art = [list(row) for row in art]
    what_lies_beneath = [[' '] * len(row) for row in art]
    art = utils.paint(art, [player_position], ['P'])
    art = utils.paint(art, [goal_position], ['G'])
    art = utils.paint(art, box_positions, [str(box_id) for box_id in box_ids])
    art = [''.join(row) for row in art]
    what_lies_beneath = [''.join(row) for row in what_lies_beneath]
    return art, what_lies_beneath, {}


def generate_pallete(num_samples, exclude=[], np_random=np.random):
    """Generate a random palette."""
    unit_vec = np_random.normal(loc=0, scale=1, size=(num_samples, 3))
    unit_vec = unit_vec / np.linalg.norm(unit_vec, ord=2, axis=-1, keepdims=True)
    unit_palette = ((unit_vec + 1) / 2) * 255.
    for exclude_palette in exclude:
        if np.all(unit_palette.astype(np.uint8) == exclude_palette.astype(np.uint8)):
            return None
    return unit_palette


goal_color = np.array([13, 171, 162])
player_color = np.array([38, 126, 218])

default_colors = {
    'G': goal_color, 
    'P': player_color,
    'W': np.array([ 13, 179, 104]),
    'B': np.array([207, 27, 128]),
    ' ': np.array([108, 133,  1]),
}

box_colors = {box: np.array([221, 205,  90]) for idx, box in enumerate(utils.BOXES)}  # same colors.
default_colors.update(box_colors)


def generate_bridge_building_colors(np_random=np.random):
    """Generate bridge building colors."""
    num_characters = len(utils.CHARACTERS_NO_G_OR_P)

    # Need to regenerate if goal or player are drawn.
    palettes = []
    for _ in range(num_characters):
        while True:
            palette = generate_pallete(
                1, 
                exclude=[goal_color, player_color], 
                np_random=np_random)
            if palette is not None:
                break
        palettes.append(palette)
    palettes = np.concatenate(palettes, axis=0)

    colors = {
        'G': goal_color, 
        'P': player_color,
        'W': palettes[0],
        'B': palettes[1],
        ' ': palettes[2],
    }
    box_colors = {box: palettes[3] for box in utils.BOXES}  # same colors.
    colors.update(box_colors)
    return colors


class BridgeBuildingEnv(pycolab_env.PyColabEnv):
    """Bridge building game."""

    def __init__(self, observation_type='layers', max_iterations=20, random_colors=False):
        merge_layer_groups = [set([str(box) for box in range(len(utils.BOXES))])]
        self.np_random = None
        
        if random_colors:
            colors = lambda: generate_bridge_building_colors(
                self.np_random if self.np_random else np.random)
        else:
            colors = default_colors

        super(BridgeBuildingEnv, self).__init__(
            game_factory=lambda: games.make_tool_use_game(
                *generate_bridge_building_art(
                    1, self.np_random if self.np_random else np.random)),
            max_iterations=max_iterations, 
            default_reward=0.,
            action_space=utils.ACTION_SPACE,
            resize_scale=32,
            observation_type=observation_type,
            delay=200,
            colors=colors,
            exclude_from_state=None if observation_type == 'rgb' else set([' ']),
            merge_layer_groups=merge_layer_groups)


if __name__ == "__main__":
    np.random.seed(42)
    env = BridgeBuildingEnv(observation_type='rgb')
    print(env._observation_order)
    env.seed(42)
    for _ in range(10):
        state = env.reset()
        for action in [2, 2, 2, 1, 1, 1, 1, 1, 3]:
            env.render()
            state, reward, _, _ = env.step(action)
            print(reward)
        env.render()
    env.close()