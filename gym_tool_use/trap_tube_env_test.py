"""Tests for trap tube environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

from gym_tool_use import trap_tube_env


class TestEnv(trap_tube_env.BaseTrapTubeEnv):

    def __init__(self,
                 art,
                 tool_position,
                 tool_size,
                 tool_direction,
                 food_position):
        self._art = art
        self._tool_position = tool_position
        self._tool_size = tool_size
        self._tool_direction = tool_direction
        self._food_position = food_position
        super(TestEnv, self).__init__(max_iterations=100)

    def _make_trap_tube_config(self):
        return trap_tube_env.TrapTubeConfig(
            art=self._art,
            tool_position=self._tool_position,
            tool_size=self._tool_size,
            tool_direction=self._tool_direction,
            food_position=self._food_position)

    def make_colors(self):
        return trap_tube_env.base_colors


class TrapTubeEnvTest(parameterized.TestCase):

    def _compare_transition(self, env, action):
        initial_state = env.reset()
        next_state, _, _, _ = env.step(action)
        return np.all(np.equal(next_state, initial_state))

    def assertNoTransition(self, env, action):
        self.assertTrue(self._compare_transition(env, action))

    def assertTransition(self, env, action):
        self.assertFalse(self._compare_transition(env, action))

    def testToolShouldNotMoveWithoutAgent(self):
        env = TestEnv(
            art=[
                '          ',
                '          ',
                '          ',
                '  mmmmmm  ',
                '  u    n  ',
                '  u    n  ',
                ' ammmmmm  ',  # →
                '          ',
                '          ',
                '          ',
            ],
            tool_position=(4, 2),
            tool_size=4,
            tool_direction=0,
            food_position=(4, 3))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.push.right)

        env = TestEnv(
            art=[
                '          ',
                '          ',
                '          ',
                '  mmmmmm  ',
                '  u    n  ',
                '  u    n  ',
                ' ammmmmm  ',  # →
                '          ',
                '          ',
                '          ',
            ],
            tool_position=(4, 2),
            tool_size=4,
            tool_direction=0,
            food_position=(4, 4))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.push.right)

        env = TestEnv(
            art=[
                '          ',
                '          ',
                '          ',
                '  mmmmmm  ',
                '  u    n  ',
                '  u    n  ',
                '  mmmmmma ',  # ←
                '          ',
                '          ',
                '          ',
            ],
            tool_position=(4, 7),
            tool_size=4,
            tool_direction=0,
            food_position=(4, 4))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.push.left)

        env = TestEnv(
            art=[
                '          ',
                '          ',
                '          ',
                '  mmmmmm  ',
                '  u    n  ',
                '  u    n  ',
                '  mmmmmma ',  # ←
                '          ',
                '          ',
                '          ',
            ],
            tool_position=(4, 7),
            tool_size=4,
            tool_direction=0,
            food_position=(4, 6))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.push.left)

        env = TestEnv(
            art=[
                '          ',
                '          ',
                '       a  ',  # ↓
                '  mmmmmm  ',
                '  u    n  ',
                '  u    n  ',
                '  mmmmmm  ',
                '          ',
                '          ',
                '          ',
            ],
            tool_position=(3, 2),
            tool_size=6,
            tool_direction=1,
            food_position=(4, 4))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.push.down)

        env = TestEnv(
            art=[
                '          ',
                '          ',
                '       a  ',  # ↓
                '  mmmmmm  ',
                '  u    n  ',
                '  u    n  ',
                '  mmmmmm  ',
                '          ',
                '          ',
                '          ',
            ],
            tool_position=(3, 2),
            tool_size=6,
            tool_direction=1,
            food_position=(5, 4))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.push.down)

        env = TestEnv(
            art=[
                '          ',
                '          ',
                '          ',
                '  mmmmmm  ',
                '  u    n  ',
                '  u    n  ',
                '  mmmmmm  ',  # ↑
                '       a  ',
                '          ',
                '          ',
            ],
            tool_position=(6, 2),
            tool_size=6,
            tool_direction=1,
            food_position=(5, 4))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.push.up)

    def testToolShouldMoveWithAgent(self):
        env = TestEnv(
            art=[
                '          ',
                '          ',
                '          ',
                '  mmmmmm  ',
                '  u    n  ',
                '  u    n  ',
                '  mmmmmm  ',
                ' a        ',  # →
                '          ',
                '          ',
            ],
            tool_position=(4, 2),
            tool_size=4,
            tool_direction=0,
            food_position=(4, 4))
        self.assertTransition(env, trap_tube_env.ACTIONS.push.right)

        env = TestEnv(
            art=[
                '          ',
                '          ',
                '          ',
                '  mmmmmm  ',
                '  u    n  ',
                '  u    n  ',
                '  mmmmmm  ',
                '   a      ',  # →
                '          ',
                '          ',
            ],
            tool_position=(4, 2),
            tool_size=4,
            tool_direction=0,
            food_position=(4, 4))
        self.assertTransition(env, trap_tube_env.ACTIONS.pull.right)

    def testBoundariesWithTool(self):
        env = TestEnv(
            art=[
                '          ',
                '          ',
                '          ',
                '  mmmmmm  ',
                '  u    n  ',
                '  u    n  ',
                '  mmmmmm  ',
                ' a        ',  # ←
                '          ',
                '          ',
            ],
            tool_position=(4, 0),
            tool_size=4,
            tool_direction=0,
            food_position=(4, 4))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.push.left)

        env = TestEnv(
            art=[
                '          ',
                '          ',
                '          ',
                '  mmmmmm  ',
                '  u    n  ',
                '  u    n  ',
                '  mmmmmm  ',
                '    a     ',  # ←
                '          ',
                '          ',
            ],
            tool_position=(4, 0),
            tool_size=4,
            tool_direction=1,
            food_position=(4, 4))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.push.left)

        env = TestEnv(
            art=[
                '          ',
                '          ',
                '          ',
                '  mmmmmm  ',
                '  u    n  ',
                '  u    n  ',
                '  mmmmmm  ',
                'a         ',  # ←
                '          ',
                '          ',
            ],
            tool_position=(4, 1),
            tool_size=4,
            tool_direction=0,
            food_position=(4, 4))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.pull.left)

        env = TestEnv(
            art=[
                '          ',
                '          ',
                '          ',
                '  mmmmmm  ',
                '  u    n  ',
                '  u    n  ',
                '  mmmmmm  ',
                'a         ',  # ←
                '          ',
                '          ',
            ],
            tool_position=(4, 1),
            tool_size=4,
            tool_direction=1,
            food_position=(4, 4))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.pull.left)

        env = TestEnv(
            art=[
                '          ',
                '          ',
                '          ',
                '  mmmmmm  ',
                'a u    n  ',  # ↑
                '  u    n  ',
                '  mmmmmm  ',
                '          ',
                '          ',
                '          ',
            ],
            tool_position=(0, 0),
            tool_size=4,
            tool_direction=0,
            food_position=(4, 4))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.push.up)

        env = TestEnv(
            art=[
                '          ',
                'a         ',
                '          ',
                '  mmmmmm  ',
                '  u    n  ',  # ↑
                '  u    n  ',
                '  mmmmmm  ',
                '          ',
                '          ',
                '          ',
            ],
            tool_position=(0, 0),
            tool_size=4,
            tool_direction=1,
            food_position=(4, 4))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.push.up)

        env = TestEnv(
            art=[
                'a         ',
                '          ',
                '          ',
                '  mmmmmm  ',
                '  u    n  ',  # ↑
                '  u    n  ',
                '  mmmmmm  ',
                '          ',
                '          ',
                '          ',
            ],
            tool_position=(1, 0),
            tool_size=4,
            tool_direction=0,
            food_position=(4, 4))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.pull.up)

        env = TestEnv(
            art=[
                'a         ',
                '          ',
                '          ',
                '  mmmmmm  ',
                '  u    n  ',  # ↑
                '  u    n  ',
                '  mmmmmm  ',
                '          ',
                '          ',
                '          ',
            ],
            tool_position=(1, 0),
            tool_size=4,
            tool_direction=1,
            food_position=(4, 4))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.pull.up)

        env = TestEnv(
            art=[
                '          ',
                '          ',
                '          ',
                '  mmmmmm  ',
                '  u    n  ',
                'a u    n  ',  # ↓
                '  mmmmmm  ',
                '          ',
                '          ',
                '          ',
            ],
            tool_position=(6, 0),
            tool_size=4,
            tool_direction=0,
            food_position=(4, 4))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.push.down)

        env = TestEnv(
            art=[
                '          ',
                '          ',
                '          ',
                '  mmmmmm  ',
                '  u    n  ',
                '  u    n  ',  # ↓
                '  mmmmmm  ',
                '          ',
                '          ',
                'a         ',
            ],
            tool_position=(5, 0),
            tool_size=4,
            tool_direction=0,
            food_position=(4, 4))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.pull.down)

        env = TestEnv(
            art=[
                '          ',
                '          ',
                '          ',
                '  mmmmmm  ',
                '  u    n  ',
                '  u    n  ',  # ↓
                '  mmmmmm  ',
                '          ',
                '          ',
                'a         ',
            ],
            tool_position=(8, 0),
            tool_size=4,
            tool_direction=1,
            food_position=(4, 4))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.pull.down)

    def testBoundariesWithFood(self):
        env = TestEnv(
            art=[
                '          ',
                '          ',
                '          ',
                '  mmmmmm  ',
                '  u    n  ',
                '  u    n  ',  # ↓
                '  mmmmmm  ',
                'a         ',
                '          ',
                '          ',
            ],
            tool_position=(8, 0),
            tool_size=4,
            tool_direction=1,
            food_position=(9, 0))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.push.down)

        env = TestEnv(
            art=[
                '          ',
                '          ',
                '          ',
                '  mmmmmm  ',
                '  u    n  ',
                '  u    n  ',  # ↓
                '  mmmmmm  ',
                '          ',
                '          ',
                'a         ',
            ],
            tool_position=(8, 0),
            tool_size=4,
            tool_direction=1,
            food_position=(9, 1))
        self.assertNoTransition(env, trap_tube_env.ACTIONS.pull.down)


if __name__ == '__main__':
    absltest.main()
