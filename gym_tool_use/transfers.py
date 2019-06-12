"""Trap tube transfers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import itertools
import numpy as np

from gym_tool_use import trap_tube_env


def _unscale_unit(unit, scale):
    """Shifts and scales the unit.

    Args:
        unit: np.array with shape [dims].
        scale: scale of the unit.

    Returns:
        scaled unit as np.array with shape [dims].
    """
    return (((unit + 1) / 2) * scale)


def _generate_unit(dims, np_random):
    """Generate a unit vector on a hypersphere.

        X ~ Normal(0, 1)
        X /= ||X||

    Args:
        dims: the dimension of the vector.
        np_random: the np random state.

    Returns:
        np.array with shape [dims].
    """
    unit_vec = np_random.normal(loc=0, scale=1, size=[dims])
    unit_vec = unit_vec / np.linalg.norm(
        unit_vec, ord=2, axis=-1, keepdims=True)
    return unit_vec


def _paint(art, positions, characters):
    """Paint characters into their positions in the art.

    Args:
        art: the 2D list of str representing the art.
        positions: list of (x, y) with same size as characters.
        characters: list of str with same size as positions.

    Returns:
        modified art - 2D list of str.
    """
    for position, character in zip(positions, characters):
        art[int(position[0])][int(position[1])] = character
    return art


def structural_color_transfer(colors, np_random):
    """Apply transfer to the color map and return another color map.

    Args:
        colors: Dictionary mapping key name to `tuple(R, G, B)`.
        np_random: np random state.

    Returns:
        Dictionary mapping key name to `tuple(R, G, B)`.
    """
    # Force the colors to be random and unique.
    color_keys = trap_tube_env.SYMBOLIC_OBJECTS + [trap_tube_env.GROUND]
    color_keys.pop(color_keys.index(trap_tube_env.TUBE2))
    color_values = []
    exclude_color_values = [
        trap_tube_env.AGENT_COLOR, trap_tube_env.FOOD_COLOR]

    for key in color_keys:
        while True:
            value = _generate_unit(3, np_random)
            value = _unscale_unit(value, 255.)
            exit_loop = True
            for color_value in (color_values + exclude_color_values):
                if np.all(value == color_value):
                    exit_loop = False
                    break
            if exit_loop:
                break
        color_values.append(value)

    new_colors = dict(zip(color_keys, color_values))
    new_colors[trap_tube_env.TUBE2] = new_colors[trap_tube_env.TUBE1]
    colors.update(new_colors)
    return colors


def symbolic_config_transfer(config, np_random):
    """Apply transfer to the config and return another config.

    Args:
        config: TrapTubeConfig.
        np_random: np random state.

    Returns:
        TrapTubeConfig.
    """
    # Change the category of tool.
    all_objects = list(trap_tube_env.SYMBOLIC_OBJECTS)
    tool_category = np_random.choice(all_objects)

    # Don't do anything if the new category is a tool.
    if tool_category == trap_tube_env.TOOL:
        return config

    # Get the positions of the object to swap.
    art = [list(row) for row in config.art]
    new_tool_positions = np.stack(
        np.where(np.array(art) == tool_category), axis=-1)

    # Get the size and direction of new "tool"
    # If the rows are the same, the tool is pointing vertical (d=0).
    new_tool_size = len(new_tool_positions)
    if new_tool_size > 1:
        new_tool_direction = int(
            new_tool_positions[0, 0] == new_tool_positions[1, 0])
    else:
        new_tool_direction = 0
    new_tool_position = new_tool_positions[0]

    # Repaint old tool to new category.
    tool_positions = []
    tool_row, tool_col = config.tool_position
    for offset in range(config.tool_size):
        tool_positions.append(
            ((tool_row + offset, tool_col),
             (tool_row, tool_col + offset))[config.tool_direction])
    art = _paint(
        art, tool_positions, [tool_category] * len(tool_positions))

    # Repaint new tool category
    art = _paint(
        art, new_tool_positions,
        [trap_tube_env.GROUND] * len(new_tool_positions))
    art = [''.join(row) for row in art]

    return trap_tube_env.TrapTubeConfig(
        art=art,
        tool_position=new_tool_position,
        tool_size=new_tool_size,
        tool_direction=new_tool_direction,
        food_position=config.food_position,
        tool_category=tool_category)


def perceptual_config_transfer(config, np_random):
    """Apply transfer to the config and return another config.

    Args:
        config: TrapTubeConfig.
        np_random: np random state.

    Returns:
        TrapTubeConfig.
    """
    del config

    # Sample height and width of tube.
    # We sample an even width and height to ensure that traps are centered.
    min_tube_size = 3
    max_tube_size = 4
    tube_height = np_random.choice([min_tube_size, max_tube_size])
    h_by_2 = math.ceil(tube_height / 2)
    tube_width = np_random.choice([min_tube_size, max_tube_size])
    w_by_2 = math.ceil(tube_width / 2)

    height = max_tube_size * 3
    width = max_tube_size * 3
    base_art = [' ' * width] * height

    # Place tube in a random position.
    # We sample a random corner position.
    tube_rows = list(range(
        max_tube_size, height - tube_height - (max_tube_size - 1)))
    tube_cols = list(range(
        max_tube_size, width - tube_width - (max_tube_size - 1)))

    tube_corner_positions = list(
        itertools.product(
            tube_rows, tube_cols))
    tube_corner_position = tube_corner_positions[
        np_random.choice(len(tube_corner_positions))]
    tube_corner_x, tube_corner_y = tube_corner_position

    # Generate traps and exits.
    num_total_traps = 2
    num_exits = 1
    num_traps = num_total_traps - num_exits

    # Sample trap locations by side.
    total_trap_sides = np_random.choice(
        list(range(4)), size=num_total_traps, replace=False)

    trap_xs = [tube_corner_x + h_by_2 - 1]
    trap_ys = [tube_corner_y]
    if tube_height == 4:
        trap_xs.append(tube_corner_x + h_by_2)
        trap_ys.append(tube_corner_y)
    trap_west_positions = list(zip(trap_xs, trap_ys))

    trap_xs = [tube_corner_x + h_by_2 - 1]
    trap_ys = [tube_corner_y + tube_width - 1]
    if tube_height == 4:
        trap_xs.append(tube_corner_x + h_by_2)
        trap_ys.append(tube_corner_y + tube_width - 1)
    trap_east_positions = list(zip(trap_xs, trap_ys))

    trap_xs = [tube_corner_x]
    trap_ys = [tube_corner_y + w_by_2 - 1]
    if tube_width == 4:
        trap_xs.append(tube_corner_x)
        trap_ys.append(tube_corner_y + w_by_2)
    trap_north_positions = list(zip(trap_xs, trap_ys))

    trap_xs = [tube_corner_x + tube_height - 1]
    trap_ys = [tube_corner_y + w_by_2 - 1]
    if tube_width == 4:
        trap_xs.append(tube_corner_x + tube_height - 1)
        trap_ys.append(tube_corner_y + w_by_2)
    trap_south_positions = list(zip(trap_xs, trap_ys))

    # Get the orientation of the tube and traps.
    trap_side_positions = [
        (0, (trap_west_positions, trap_east_positions)),
        (1, (trap_north_positions, trap_south_positions))]
    trap_direction, new_trap_side_positions = trap_side_positions[
        np_random.choice(2)]
    exit_index = np_random.choice(2)
    exit_positions = new_trap_side_positions[exit_index]
    trap_positions = new_trap_side_positions[1 - exit_index]

    # Get the tube1, tube2 positions.
    tube_row_range = range(tube_corner_x, tube_corner_x + tube_height)
    tube_col_range = range(tube_corner_y, tube_corner_y + tube_width)
    west_tube_positions = [(x, tube_corner_y) for x in tube_row_range]
    east_tube_positions = [
        (x, tube_corner_y + tube_width - 1) for x in tube_row_range]
    north_tube_positions = [(tube_corner_x, y) for y in tube_col_range]
    south_tube_positions = [
        (tube_corner_x + tube_height - 1, y) for y in tube_col_range]
    if trap_direction == 1:
        tube_positions = [west_tube_positions, east_tube_positions]
    elif trap_direction == 0:
        tube_positions = [north_tube_positions, south_tube_positions]

    tube1_index = np_random.choice(2)
    tube2_index = 1 - tube1_index
    tube1_positions = tube_positions[tube1_index]
    tube2_positions = tube_positions[tube2_index]

    # Place food in the tube.
    # We sample a random position within the trap tube.
    food_rows = list(range(
        tube_corner_x + 1, tube_corner_x + tube_height - 1))
    food_cols = list(range(
        tube_corner_y + 1, tube_corner_y + tube_width - 1))
    food_positions = list(
        itertools.product(
            food_rows, food_cols))
    food_position = food_positions[
        np_random.choice(len(food_positions))]

    # Place tool in a random position and direction.
    # Sample random tool direction, set the size equal to the tube width
    # or height depending on the direction.
    tool_direction = np_random.choice(2)
    tool_size, row_offset, col_offset = (
        (tube_height, tube_height, 0), (tube_width, 0, tube_width))[
            tool_direction]
    tool_rows = list(range(0, height - row_offset))
    tool_cols = list(range(0, width - col_offset))
    tool_positions = list(
        itertools.product(
            tool_rows, tool_cols))

    # Remove the tube area from the possible tool positions.
    tube_rows = list(range(tube_corner_x, tube_corner_x + tube_height))
    tube_cols = list(range(tube_corner_y, tube_corner_y + tube_width))
    tube_area_positions = list(
        itertools.product(
            tube_rows, tube_cols))
    for remove_position in tube_area_positions:
        if remove_position in tool_positions:
            tool_positions.remove(remove_position)

    # Remove the areas around the tube area.
    if tool_direction == 0:
        area_rows = list(range(
            tube_corner_x - tube_height, tube_corner_x + tube_height))
        area_cols = list(range(
            tube_corner_y - 2, tube_corner_y + tube_width + 2))
        area_positions = list(
            itertools.product(
                area_rows, area_cols))
        for remove_position in area_positions:
            if remove_position in tool_positions:
                tool_positions.remove(remove_position)

    if tool_direction == 1:
        area_rows = list(range(
            tube_corner_x - 2, tube_corner_x + tube_height + 2))
        area_cols = list(range(
            tube_corner_y - tube_width, tube_corner_y + tube_width))
        area_positions = list(
            itertools.product(
                area_rows, area_cols))
        for remove_position in area_positions:
            if remove_position in tool_positions:
                tool_positions.remove(remove_position)

    # Generate the tool position.
    tool_position = tool_positions[
        np_random.choice(len(tool_positions))]

    # Start the agent in a random position around the tube.
    agent_positions = []
    agent_rows = list(range(0, height))
    agent_cols = list(range(0, width))
    agent_positions = list(
        itertools.product(
            agent_rows, agent_cols))
    tube_area_rows = list(range(max_tube_size, max_tube_size * 2 + 1))
    tube_area_cols = list(range(max_tube_size, max_tube_size * 2 + 1))
    for tube_area_position in itertools.product(
            tube_area_rows, tube_area_cols):
        agent_positions.remove(tube_area_position)
    for position in tool_positions:
        if position in agent_positions:
            agent_positions.remove(position)
    agent_position = agent_positions[
        np_random.choice(len(agent_positions))]

    art = [list(row) for row in base_art]
    art = _paint(
        art, tube1_positions, [trap_tube_env.TUBE1] * len(tube1_positions))
    art = _paint(
        art, tube2_positions, [trap_tube_env.TUBE2] * len(tube2_positions))
    art = _paint(
        art, exit_positions,
        [trap_tube_env.EXIT] * len(exit_positions))
    art = _paint(
        art, trap_positions,
        [trap_tube_env.TRAP] * len(trap_positions))
    art = _paint(art, [agent_position], [trap_tube_env.AGENT])
    art = [''.join(row) for row in art]

    return trap_tube_env.TrapTubeConfig(
        art=art,
        tool_position=tool_position,
        tool_size=tool_size,
        tool_direction=tool_direction,
        food_position=food_position,
        tool_category=trap_tube_env.TOOL)


class BaseTransferTrapTubeEnv(trap_tube_env.BaseTrapTubeEnv):

    def __init__(self,
                 config_transfers,
                 color_transfers,
                 initial_config,
                 initial_colors,
                 max_iterations=100):
        """Creates a new BaseTransferTrapTubeEnv.

        Forms a base for all trap transfer environments.

        Args:
            config_transfers: list of functions to apply to the configs.
            color_transfers: list of functions to apply to the colors.
            initial_config: TrapTubeConfig.
            initial_colors: Dictionary mapping key name to `tuple(R, G, B)`.
            max_iterations: maximum number of steps allowed.
        """
        self._config_transfers = config_transfers
        self._color_transfers = color_transfers
        self._initial_config = initial_config
        self._initial_colors = initial_colors
        self._tool_category = trap_tube_env.TOOL
        super(BaseTransferTrapTubeEnv, self).__init__(
            max_iterations=max_iterations)

    def _make_trap_tube_config(self):
        np_random = self.np_random if self.np_random else np.random
        config = self._initial_config
        for transfer in self._config_transfers:
            config = transfer(config, np_random)
        self._tool_category = config.tool_category
        return config

    def make_colors(self):
        np_random = self.np_random if self.np_random else np.random
        colors = self._initial_colors
        for transfer in self._color_transfers:
            colors = transfer(colors, np_random)
        # swap tool colors.
        if self._tool_category is not trap_tube_env.TOOL:
            new_tool_color = colors[self._tool_category]
            colors[self._tool_category] = colors[trap_tube_env.TOOL]
            colors[trap_tube_env.TOOL] = new_tool_color
        return colors


class PerceptualTrapTubeEnv(BaseTransferTrapTubeEnv):

    def __init__(self, max_iterations=100):
        super(PerceptualTrapTubeEnv, self).__init__(
            config_transfers=[perceptual_config_transfer],
            color_transfers=[],
            initial_config=trap_tube_env.base_config,
            initial_colors=dict(trap_tube_env.base_colors),
            max_iterations=max_iterations)


class StructuralTrapTubeEnv(BaseTransferTrapTubeEnv):

    def __init__(self, max_iterations=100):
        super(StructuralTrapTubeEnv, self).__init__(
            config_transfers=[],
            color_transfers=[structural_color_transfer],
            initial_config=trap_tube_env.base_config,
            initial_colors=dict(trap_tube_env.base_colors),
            max_iterations=max_iterations)


class SymbolicTrapTubeEnv(BaseTransferTrapTubeEnv):

    def __init__(self, max_iterations=100):
        super(SymbolicTrapTubeEnv, self).__init__(
            config_transfers=[symbolic_config_transfer],
            color_transfers=[],
            initial_config=trap_tube_env.base_config,
            initial_colors=dict(trap_tube_env.base_colors),
            max_iterations=max_iterations)


class StructuralSymbolicTrapTubeEnv(BaseTransferTrapTubeEnv):

    def __init__(self, max_iterations=100):
        super(StructuralSymbolicTrapTubeEnv, self).__init__(
            config_transfers=[symbolic_config_transfer],
            color_transfers=[
                structural_color_transfer],
            initial_config=trap_tube_env.base_config,
            initial_colors=dict(trap_tube_env.base_colors),
            max_iterations=max_iterations)


class PerceptualStructuralTrapTubeEnv(BaseTransferTrapTubeEnv):

    def __init__(self, max_iterations=100):
        super(PerceptualStructuralTrapTubeEnv, self).__init__(
            config_transfers=[perceptual_config_transfer],
            color_transfers=[structural_color_transfer],
            initial_config=trap_tube_env.base_config,
            initial_colors=dict(trap_tube_env.base_colors),
            max_iterations=max_iterations)


class PerceptualSymbolicTrapTubeEnv(BaseTransferTrapTubeEnv):

    def __init__(self, max_iterations=100):
        super(PerceptualSymbolicTrapTubeEnv, self).__init__(
            config_transfers=[
                perceptual_config_transfer, symbolic_config_transfer],
            color_transfers=[],
            initial_config=trap_tube_env.base_config,
            initial_colors=dict(trap_tube_env.base_colors),
            max_iterations=max_iterations)


class PerceptualStructuralSymbolicTrapTubeEnv(BaseTransferTrapTubeEnv):

    def __init__(self, max_iterations=100):
        super(PerceptualStructuralSymbolicTrapTubeEnv, self).__init__(
            config_transfers=[
                perceptual_config_transfer, symbolic_config_transfer],
            color_transfers=[
                structural_color_transfer],
            initial_config=trap_tube_env.base_config,
            initial_colors=dict(trap_tube_env.base_colors),
            max_iterations=max_iterations)


class TrapTubeEnv(trap_tube_env.BaseTrapTubeEnv):

    def _make_trap_tube_config(self):
        return trap_tube_env.base_config

    def make_colors(self):
        return trap_tube_env.base_colors


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--transfer',
        choices=[
            'p',
            'st',
            'sy',
            'stsy',
            'pst',
            'psy',
            'pstsy',
            'n',
        ],
        required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.transfer == 'p':
        constructor = PerceptualTrapTubeEnv
    elif args.transfer == 'st':
        constructor = StructuralTrapTubeEnv
    elif args.transfer == 'sy':
        constructor = SymbolicTrapTubeEnv
    elif args.transfer == 'stsy':
        constructor = StructuralSymbolicTrapTubeEnv
    elif args.transfer == 'pst':
        constructor = PerceptualStructuralTrapTubeEnv
    elif args.transfer == 'psy':
        constructor = PerceptualSymbolicTrapTubeEnv
    elif args.transfer == 'pstsy':
        constructor = PerceptualStructuralSymbolicTrapTubeEnv
    elif args.transfer == 'n':
        constructor = TrapTubeEnv

    for i in range(100):
        env = constructor()
        env.seed(i + args.seed)
        state = env.reset()
        env.render()
        total_reward = 0.
        for _ in range(10):
            state, reward, done, _ = env.step(env.action_space.sample())
            env.render()
            total_reward += reward
            if done:
                break
        env.close()
        print(total_reward)
