from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


def structural_transfer(colors, np_random):
    """Apply transfer to the color map and return another color map.

    Args:
        colors: Dictionary mapping key name to `tuple(R, G, B)`.
        np_random: np random state.

    Returns:
        Dictionary mapping key name to `tuple(R, G, B)`.
    """
    # Force the colors to be random and unique.
    color_keys = [
        trap_tube_env.TOOL, trap_tube_env.TUBE,
        trap_tube_env.TRAP, trap_tube_env.FAKE_TRAP,
        trap_tube_env.GROUND]
    color_values = []
    for key in color_keys:
        while True:
            value = _generate_unit(3, np_random)
            value = _unscale_unit(value, 255.)
            exit_loop = True
            for color_value in color_values:
                if np.all(value == color_value):
                    exit_loop = False
                    break
            if exit_loop:
                break
        color_values.append(value)

    new_colors = dict(zip(color_keys, color_values))
    colors.update(new_colors)
    return colors


def symbolic_transfer(colors, np_random):
    """Apply transfer to the color map and return another color map.

    Args:
        colors: Dictionary mapping key name to `tuple(R, G, B)`.
        np_random: np random state.

    Returns:
        Dictionary mapping key name to `tuple(R, G, B)`.
    """
    new_colors = dict(colors)
    all_objects = list(trap_tube_env.SYMBOLIC_OBJECTS)
    indices = np.random.choice(len(all_objects), size=2, replace=False)
    gt_objects = []
    for index in indices:
        gt_objects.append(all_objects[index])
    for obj in gt_objects:
        all_objects.remove(obj)
    for obj in all_objects:
        new_colors[obj] = new_colors[np.random.choice(gt_objects)]
    return new_colors


def perceptual_transfer(config, np_random):
    """Apply transfer to the config and return another config.

    Args:
        config: TrapTubeConfig.
        np_random: np random state.

    Returns:
        TrapTubeConfig.
    """
    del config

    base_art = [
        '          ',
        '          ',
        '          ',
        '          ',
        '          ',
        '          ',
        '          ',
        '          ',
        '          ',
        '          ',
    ]
    height = len(base_art)
    width = len(base_art[0])

    # Sample height and width of tube.
    # We sample an even width and height to ensure that traps are centered.
    tube_height = np_random.choice([4, 6])
    h_by_2 = tube_height / 2
    tube_width = np_random.choice([4, 6])
    w_by_2 = tube_width / 2

    # Place tube in a random position.
    # We sample a random corner position.
    tube_rows = list(range(1, height - tube_height - 1))
    tube_cols = list(range(1, width - tube_width - 1))
    tube_corner_positions = list(
        itertools.product(
            tube_rows, tube_cols))
    tube_corner_position = tube_corner_positions[
        np_random.choice(len(tube_corner_positions))]
    tube_corner_x, tube_corner_y = tube_corner_position

    # Generate traps and fake traps.
    # We can sample from 1 to 4 traps.
    num_total_traps = np_random.choice(range(1, 5))
    num_fake_traps = 1
    num_traps = num_total_traps - num_fake_traps

    # Sample trap locations by side.
    total_trap_sides = np_random.choice(
        list(range(4)), size=num_total_traps, replace=False)
    total_trap_positions = [
        # West
        list(
            zip([tube_corner_x + h_by_2 - 1, tube_corner_x + h_by_2],
                [tube_corner_y, tube_corner_y])),

        # East
        list(
            zip([tube_corner_x + h_by_2 - 1, tube_corner_x + h_by_2],
                [tube_corner_y + tube_width - 1,
                tube_corner_y + tube_width - 1])),

        # North
        list(
            zip([tube_corner_x, tube_corner_x],
                [tube_corner_y + w_by_2 - 1, tube_corner_y + w_by_2])),

        # South
        list(
            zip([tube_corner_x + tube_height - 1,
                 tube_corner_x + tube_height - 1],
                [tube_corner_y + w_by_2 - 1, tube_corner_y + w_by_2])),
    ]
    fake_trap_positions = total_trap_positions[total_trap_sides[0]]
    trap_positions = []
    for trap_side in total_trap_sides[1:]:
        trap_positions.extend(total_trap_positions[trap_side])

    tube_positions = []
    tube_row_range = range(tube_corner_x, tube_corner_x + tube_height)
    tube_col_range = range(tube_corner_y, tube_corner_y + tube_width)

    # West
    tube_positions += [(x, tube_corner_y) for x in tube_row_range]

    # East
    tube_positions += [
        (x, tube_corner_y + tube_width - 1) for x in tube_row_range]

    # North
    tube_positions += [(tube_corner_x, y) for y in tube_col_range]

    # South
    tube_positions += [
        (tube_corner_x + tube_height - 1, y) for y in tube_col_range]

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
            tube_rows, tube_cols))
    if food_position in tool_positions:
        tool_positions.remove(food_position)
    tool_position = tool_positions[
        np_random.choice(len(tool_positions))]

    # Always start agents on the edges.
    agent_positions = []
    agent_row_range = list(range(0, height))
    agent_col_range = list(range(0, width))

    # West
    agent_positions += [(x, 0) for x in agent_row_range]

    # East
    agent_positions += [(x, width - 1) for x in agent_row_range]

    # North
    agent_positions += [(0, y) for y in agent_col_range]

    # South
    agent_positions += [(height - 1, y) for y in agent_col_range]

    for position in tool_positions:
        if position in agent_positions:
            agent_positions.remove(position)
    agent_position = agent_positions[
        np_random.choice(len(agent_positions))]

    art = [list(row) for row in base_art]
    art = _paint(
        art, tube_positions, [trap_tube_env.TUBE] * len(tube_positions))

    art = _paint(
        art, fake_trap_positions,
        [trap_tube_env.FAKE_TRAP] * len(fake_trap_positions))
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
        food_position=food_position)


class BaseTransferTrapTubeEnv(trap_tube_env.BaseTrapTubeEnv):

    def __init__(self,
                 config_transfers,
                 color_transfers,
                 initial_config,
                 initial_colors,
                 max_iterations=50):
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
        super(BaseTransferTrapTubeEnv, self).__init__(
            max_iterations=max_iterations)

    def _make_trap_tube_config(self):
        """Create the game art.

        Returns:
            TrapTubeConfig.
        """
        np_random = self.np_random if self.np_random else np.random
        config = self._initial_config
        for transfer in self._config_transfers:
            config = transfer(config, np_random)
        return config

    def make_colors(self):
        np_random = self.np_random if self.np_random else np.random
        colors = self._initial_colors
        for transfer in self._color_transfers:
            colors = transfer(colors, np_random)
        return colors


class PerceptualTrapTubeEnv(BaseTransferTrapTubeEnv):

    def __init__(self, max_iterations=50):
        super(PerceptualTrapTubeEnv, self).__init__(
            config_transfers=[perceptual_transfer],
            color_transfers=[],
            initial_config=trap_tube_env.base_config,
            initial_colors=dict(trap_tube_env.base_colors),
            max_iterations=max_iterations)


class StructuralTrapTubeEnv(BaseTransferTrapTubeEnv):

    def __init__(self, max_iterations=50):
        super(StructuralTrapTubeEnv, self).__init__(
            config_transfers=[],
            color_transfers=[structural_transfer],
            initial_config=trap_tube_env.base_config,
            initial_colors=dict(trap_tube_env.base_colors),
            max_iterations=max_iterations)


class SymbolicTrapTubeEnv(BaseTransferTrapTubeEnv):

    def __init__(self, max_iterations=50):
        super(SymbolicTrapTubeEnv, self).__init__(
            config_transfers=[],
            color_transfers=[symbolic_transfer],
            initial_config=trap_tube_env.base_config,
            initial_colors=dict(trap_tube_env.base_colors),
            max_iterations=max_iterations)


class StructuralSymbolicTrapTubeEnv(BaseTransferTrapTubeEnv):

    def __init__(self, max_iterations=50):
        super(StructuralSymbolicTrapTubeEnv, self).__init__(
            config_transfers=[],
            color_transfers=[structural_transfer, symbolic_transfer],
            initial_config=trap_tube_env.base_config,
            initial_colors=dict(trap_tube_env.base_colors),
            max_iterations=max_iterations)


class PerceptualStructuralTrapTubeEnv(BaseTransferTrapTubeEnv):

    def __init__(self, max_iterations=50):
        super(PerceptualStructuralTrapTubeEnv, self).__init__(
            config_transfers=[perceptual_transfer],
            color_transfers=[structural_transfer],
            initial_config=trap_tube_env.base_config,
            initial_colors=dict(trap_tube_env.base_colors),
            max_iterations=max_iterations)


class PerceptualSymbolicTrapTubeEnv(BaseTransferTrapTubeEnv):

    def __init__(self, max_iterations=50):
        super(PerceptualSymbolicTrapTubeEnv, self).__init__(
            config_transfers=[perceptual_transfer],
            color_transfers=[symbolic_transfer],
            initial_config=trap_tube_env.base_config,
            initial_colors=dict(trap_tube_env.base_colors),
            max_iterations=max_iterations)


class PerceptualStructuralSymbolicTrapTubeEnv(BaseTransferTrapTubeEnv):

    def __init__(self, max_iterations=50):
        super(PerceptualStructuralSymbolicTrapTubeEnv, self).__init__(
            config_transfers=[perceptual_transfer],
            color_transfers=[structural_transfer, symbolic_transfer],
            initial_config=trap_tube_env.base_config,
            initial_colors=dict(trap_tube_env.base_colors),
            max_iterations=max_iterations)


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
        ],
        required=True)
    args = parser.parse_args()
    np.random.seed(42)

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

    for _ in range(10):
        env = constructor()
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
