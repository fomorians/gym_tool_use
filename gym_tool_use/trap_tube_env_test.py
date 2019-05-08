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
        super(TestEnv, self).__init__(
            max_iterations=100,
            delay=240)

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

    def setUp(self):
        super(TrapTubeEnvTest, self).setUp()
        self._render = True

    def _compare_transition(self, env, action, render=False):
        initial_state = env.reset()
        if render:
            env.render()
        next_state, _, _, _ = env.step(action)
        if render:
            env.render()
            env.close()
        return np.all(np.equal(next_state, initial_state))

    def assertNoTransition(self, env, action, render=False):
        self.assertTrue(
            self._compare_transition(env, action, render=render))

    def assertTransition(self, env, action, render=False):
        self.assertFalse(
            self._compare_transition(env, action, render=render))

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
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.push.right, render=self._render)

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
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.push.right, render=self._render)

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
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.push.left, render=self._render)

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
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.push.left, render=self._render)

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
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.push.down, render=self._render)

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
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.push.down, render=self._render)

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
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.push.up, render=self._render)

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
        self.assertTransition(
            env, trap_tube_env.ACTIONS.push.right, render=self._render)

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
        self.assertTransition(
            env, trap_tube_env.ACTIONS.pull.right, render=self._render)

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
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.push.left, render=self._render)

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
            tool_position=(7, 0),
            tool_size=4,
            tool_direction=1,
            food_position=(4, 4))
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.push.left, render=self._render)

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
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.pull.left, render=self._render)

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
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.pull.left, render=self._render)

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
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.push.up, render=self._render)

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
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.push.up, render=self._render)

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
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.pull.up, render=self._render)

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
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.pull.up, render=self._render)

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
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.push.down, render=self._render)

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
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.pull.down, render=self._render)

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
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.pull.down, render=self._render)

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
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.push.down, render=self._render)

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
        self.assertNoTransition(
            env, trap_tube_env.ACTIONS.pull.down, render=self._render)


if __name__ == '__main__':
    absltest.main()
