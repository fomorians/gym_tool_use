"""The Bridge Building Environment."""

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
    '        ',  # WWWWWWWW
    '        ', 
    '        ', 
    '        ',
]
DIVIDER_X = 4


def generate_bridge_building_art(randomly_swap_water_and_boxes=False, 
                                 np_random=np.random):
    """Generate bridge building art."""
    num_x = num_y = len(BRIDGE_BUILDING_TEMPLATE)
    c_l, c_r = 2, num_y - 2

    # Generate player position
    player_y_position = np_random.randint(c_l, c_r)
    player_x_position = np_random.randint(0, DIVIDER_X)
    player_position = (player_x_position, player_y_position)

    # Generate goal position
    goal_y_position = np_random.randint(c_l, c_r)
    goal_x_position = np_random.randint(DIVIDER_X + 1, num_x)
    goal_position = (goal_x_position, goal_y_position)

    # TODO(wenkesj): is this necessary now?
    # Generate box positions out of the shortest path.
    c_l_range = list(range(c_l - 1, min(player_y_position, goal_y_position)))
    c_r_range = list(range(max(player_y_position, goal_y_position) + 1, c_r + 1))
    box_side_positions = [
        (x, y)
        for x in range(1, DIVIDER_X)  # can't start in the first row,
                                      # nor in the river.
        for y in c_l_range + c_r_range]

    if player_position in box_side_positions:
        box_side_positions.pop(box_side_positions.index(player_position))

    box_position_indices = np_random.choice(
        len(box_side_positions), size=1, replace=False)
    box_positions = [box_side_positions[box_position_index] 
                    for box_position_index in box_position_indices]

    # Assign random numbers to the boxes.
    box_ids = np_random.choice(len(utils.BOXES), size=1, replace=False)

    # Assign a river.
    water_positions = [(DIVIDER_X, y) for y in range(num_y)]

    # Paint positions
    art = list(BRIDGE_BUILDING_TEMPLATE)
    art = [list(row) for row in art]
    what_lies_beneath = [[' '] * len(row) for row in art]
    art = utils.paint(art, [player_position], ['P'])
    art = utils.paint(art, [goal_position], ['G'])

    # Randomly swap between painting water or box tiles.
    water_ids = ['W'] * num_x
    box_ids = [str(box_id) for box_id in box_ids]

    if randomly_swap_water_and_boxes:
        # Swap "uniformly discrete".
        if np_random.randint(2):
            print('swapping boxes.')
            tmp_water_positions = list(water_positions)
            # Generate enough boxes to fill the "river".
            water_positions = box_positions
            water_ids = ['W'] * len(box_positions)

            box_positions = tmp_water_positions
            box_ids = [str(box_id) for box_id in range(num_y)]
            print(box_ids)

    art = utils.paint(art, water_positions, water_ids)
    art = utils.paint(art, box_positions, box_ids)

    art = [''.join(row) for row in art]
    what_lies_beneath = [''.join(row) for row in what_lies_beneath]
    return art, what_lies_beneath, {}


def generate_color(exclude=[], np_random=np.random):
    """Generate a random color on a hypersphere."""
    unit_vec = np_random.normal(loc=0, scale=1, size=(1, 3))
    unit_vec = unit_vec / np.linalg.norm(unit_vec, ord=2, axis=-1, keepdims=True)
    unit_palette = ((unit_vec + 1) / 2) * 255.  # normalize and scale to color space.
    for exclude_palette in exclude:
        if np.all(unit_palette.astype(np.uint8) == exclude_palette.astype(np.uint8)):
            return None
    return unit_palette


goal_color = np.array([13, 171, 162])
player_color = np.array([38, 126, 218])
bg_color = np.array([108, 133,  1])

default_colors = {
    'G': goal_color, 
    'P': player_color,
    'W': np.array([ 13, 179, 104]),
    'B': np.array([207, 27, 128]),
    ' ': bg_color,
}

box_colors = {box: np.array([221, 205,  90]) for idx, box in enumerate(utils.BOXES)}  # same colors.
default_colors.update(box_colors)


def generate_bridge_building_colors(np_random=np.random):
    """Generate bridge building colors."""

    # Need to regenerate if goal or player are drawn.
    palette = []
    for _ in range(3):
        while True:
            color = generate_color(
                exclude=[goal_color, player_color] + palette, 
                np_random=np_random)
            if color is not None:
                break
        palette.append(color[0])
    palette = np.stack(palette, axis=0)

    colors = {
        'G': goal_color, 
        'P': player_color,
        'W': palette[0],
        'B': palette[1],
        ' ': bg_color,
    }
    box_colors = {box: palette[2] for box in utils.BOXES}  # same colors.
    colors.update(box_colors)
    return colors


class BridgeBuildingEnv(pycolab_env.PyColabEnv):
    """Bridge building game."""

    def __init__(self, 
                 observation_type='layers', 
                 max_iterations=25,
                 random_colors=False, 
                 randomly_swap_water_and_boxes=False,
                 override_art=None):
        merge_layer_groups = [set([str(box) for box in range(len(utils.BOXES))])]
        self.np_random = None

        if random_colors:
            colors = lambda: generate_bridge_building_colors(
                self.np_random if self.np_random else np.random)
        else:
            colors = default_colors

        if override_art:
            generate_art = lambda: (override_art, ' ', {})
        else:
            generate_art = lambda: generate_bridge_building_art(
                randomly_swap_water_and_boxes=randomly_swap_water_and_boxes,
                np_random=self.np_random if self.np_random else np.random)

        super(BridgeBuildingEnv, self).__init__(
            game_factory=lambda: games.make_tool_use_game(*generate_art()),
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