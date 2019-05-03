from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from gym_tool_use import trap_tube_env


class SymbolicTrapTubeEnv(trap_tube_env.BaseTrapTubeEnv):

    def __init__(self,
                 colors,
                 tool_position,
                 tool_size,
                 tool_direction,
                 food_position,
                 max_iterations=50):
        """Creates a new SymbolicTrapTubeEnv.

        This trap tube configuration tests an agents ability to generalize to
        objects that no longer have guarenteed perceptual or structural
        representations. Whilst increasing the salience of a cue should
        facilitate learning at the perceptual level, conversely, it is likely
        to interfere at the symbolic level, because increased appreciation of
        the object itself may block an appreciation of its symbolic role
        (DeLoache 2004).

        Traps are transparent and the agent must first discover that the trap
        exists and must use the tool to move the food. If the agent percieves
        that the trap does not exist, it will have trouble solving the task.

        Symbolic generalization attempts to test the agents ability to
        regularize it's perceptual and structural representations of the world.

        Args:
            colors: Dictionary mapping key name to `tuple(R, G, B)`.
            tool_position: position of the tool `(row, col)` of the top of the
                tool.
            tool_size: the size of the tool, spanning range in tool_direction.
            tool_direction: the direction of the tool, 0 or 1.
            food_position: the position of the food.
            max_iterations: maximum number of steps allowed.
        """
        self._colors = colors
        self._tool_position = tool_position
        self._tool_size = tool_size
        self._tool_direction = tool_direction
        self._food_position = food_position
        super(SymbolicTrapTubeEnv, self).__init__(
            max_iterations=max_iterations)

    def _make_trap_tube_config(self):
        """Creates a single example of structural generalization.

        All episodes of a level have the _same_ solutions. There are no
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
        if trap_tube_env.FOOD not in self._colors:
            self._colors[trap_tube_env.FOOD] = trap_tube_env.FOOD_COLOR
        if trap_tube_env.AGENT not in self._colors:
            self._colors[trap_tube_env.AGENT] = trap_tube_env.AGENT_COLOR

        # Force the symbolism of the traps.
        # We set the trap and fake trap colors to the ground color.
        self._colors[trap_tube_env.TRAP] = self._colors[trap_tube_env.GROUND]
        self._colors[trap_tube_env.FAKE_TRAP] = self._colors[
            trap_tube_env.GROUND]
        return self._colors


if __name__ == '__main__':
    np.random.seed(42)

    for _ in range(10):
        env = SymbolicTrapTubeEnv(
            colors={
                trap_tube_env.TOOL:      (152, 208, 57),
                trap_tube_env.FAKE_TOOL: (152, 255, 57),
                trap_tube_env.TUBE:      (57, 152, 208),
                trap_tube_env.FAKE_TUBE: (57, 202, 208),
                trap_tube_env.TRAP:      (208, 113, 57),
                trap_tube_env.FAKE_TRAP: (208, 189, 57),
                trap_tube_env.GROUND:    (72, 65, 17),
            },
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
