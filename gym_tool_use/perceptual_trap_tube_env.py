from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np

from gym_tool_use import trap_tube_env


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


class PerceptualTrapTubeEnv(trap_tube_env.BaseTrapTubeEnv):

    def __init__(self, colors, max_iterations=50):
        """Creates a new PerceptualTrapTubeEnv.

        This trap tube configuration tests an agents ability to
        generalize to perceptual invariants. A subject reliant
        on perceptual knowledge based on the visual appearance
        of the task would not be expected to be able to solve
        the problem in the dark [...] (Seed 2011).

        Features are shared across episodes, but the positions
        and shapes are randomly sampled.

        Args:
            colors: Dictionary mapping key name to `tuple(R, G, B)`.
            max_iterations: maximum number of steps allowed.
        """
        self._colors = colors
        super(PerceptualTrapTubeEnv, self).__init__(
            max_iterations=max_iterations)

    def _make_trap_tube_config(self):
        """Creates a single example of perceptual generalization.

        All episodes of a level have _different_ solutions. There are no
        structural changes, only perceptual.

        Returns:
            trap_tube_env.TrapTubeConfig
        """
        np_random = self.np_random if self.np_random else np.random

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
        # TODO(wenkesj): 8 is probably too much.
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
            fake_tool_position=(-1, -1),
            fake_tool_size=0,
            fake_tool_direction=0,
            food_position=food_position)

    def make_colors(self):
        if trap_tube_env.FOOD not in self._colors:
            self._colors[trap_tube_env.FOOD] = trap_tube_env.FOOD_COLOR
        if trap_tube_env.AGENT not in self._colors:
            self._colors[trap_tube_env.AGENT] = trap_tube_env.AGENT_COLOR
        return self._colors


if __name__ == '__main__':
    np.random.seed(42)

    for _ in range(10):
        env = PerceptualTrapTubeEnv({
            trap_tube_env.TOOL:      (152, 208, 57),
            trap_tube_env.FAKE_TOOL: (152, 255, 57),
            trap_tube_env.TUBE:      (57, 152, 208),
            trap_tube_env.FAKE_TUBE: (57, 202, 208),
            trap_tube_env.TRAP:      (208, 113, 57),
            trap_tube_env.FAKE_TRAP: (208, 189, 57),
            trap_tube_env.GROUND:    (72, 65, 17),
        })
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
