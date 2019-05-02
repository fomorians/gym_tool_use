from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


class StructuralTrapTubeEnv(trap_tube_env.BaseTrapTubeEnv):

    def __init__(self,
                 tool_position,
                 tool_size,
                 tool_direction,
                 food_position,
                 sample_agent_color=False,
                 sample_food_color=False,
                 max_iterations=50):
        """Creates a new StructuralTrapTubeEnv.

        This trap tube configuration tests an agents ability to generalize to
        feature representations. Generalization of the solution should
        therefore be possible across various changes to the perceptual elements
        of the original input, as long as the causal logic was unchanged
        (Seed 2011).

        Features are functionally related to each other. We sample
        multi-modal features (represented by RGB colors) as unit-vectors on a
        hypersphere.

        Can agents learn to ignore arbitrary feature representations to solve
        the task?

        Args:
            tool_position: position of the tool `(row, col)` of the top of the
                tool.
            tool_size: the size of the tool, spanning range in tool_direction.
            tool_direction: the direction of the tool, 0 or 1.
            food_position: the position of the food.
            sample_agent_color: flag to allow agents color to be sampled.
                `False` by default.
            sample_food_color: flag to allow food color to be sampled. `False`
                by default.
            max_iterations: maximum number of steps allowed.
        """
        self._tool_position = tool_position
        self._tool_size = tool_size
        self._tool_direction = tool_direction
        self._food_position = food_position
        self._sample_agent_color = sample_agent_color
        self._sample_food_color = sample_food_color
        super(StructuralTrapTubeEnv, self).__init__(
            max_iterations=max_iterations)

    def _make_trap_tube_config(self):
        """Creates a single example of structural generalization.

        All episodes of a level has the _same_ solutions. There are no
        perceptual changes, only structural via feature transformations.

        Returns:
            trap_tube_env.TrapTubeConfig
        """
        art = [
            '          ',
            '          ',
            '          ',
            '  mmmmmm  ',
            '  u    n  ',
            '  u    n  ',
            '  mmmmmm  ',
            '          ',
            'a         ',
            '          ',
        ]
        return trap_tube_env.TrapTubeConfig(
            art=art,
            tool_position=self._tool_position,
            tool_size=self._tool_size,
            tool_direction=self._tool_direction,
            fake_tool_position=(-1, -1),
            fake_tool_size=0,
            fake_tool_direction=0,
            food_position=self._food_position)

    def make_colors(self):
        """Samples a unique color for each class of object in the environment.

        Samples are represented as a unit vector on a hypersphere scaled
        to [0, 255].

        Returns:
            Dictionary mapping key name to `tuple(R, G, B)`.
        """
        np_random = self.np_random if self.np_random else np.random
        color_keys = [
            trap_tube_env.TOOL, trap_tube_env.TUBE,
            trap_tube_env.TRAP, trap_tube_env.FAKE_TRAP,
            trap_tube_env.GROUND]
        color_values = []

        if self._sample_food_color:
            color_keys.append(trap_tube_env.FOOD)
        if self._sample_agent_color:
            color_keys.append(trap_tube_env.AGENT)

        for key in color_keys:
            # We sample _unique_ colors.
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

        colors = dict(zip(color_keys, color_values))
        if not self._sample_food_color:
            colors[trap_tube_env.FOOD] = trap_tube_env.FOOD_COLOR
        if not self._sample_agent_color:
            colors[trap_tube_env.AGENT] = trap_tube_env.AGENT_COLOR
        return colors


if __name__ == '__main__':
    np.random.seed(42)

    for _ in range(10):
        env = StructuralTrapTubeEnv(
            tool_position=(3, 3),
            tool_size=4,
            tool_direction=0,
            food_position=(4, 4))
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
