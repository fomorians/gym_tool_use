"""Tests for trap tube environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import unittest

from absl.testing import absltest
from absl.testing import parameterized

from gym_tool_use import trap_tube_env


uu = trap_tube_env.ACTIONS.up.up
ud = trap_tube_env.ACTIONS.up.down
ul = trap_tube_env.ACTIONS.up.left
ur = trap_tube_env.ACTIONS.up.right
du = trap_tube_env.ACTIONS.down.up
dd = trap_tube_env.ACTIONS.down.down
dl = trap_tube_env.ACTIONS.down.left
dr = trap_tube_env.ACTIONS.down.right
lu = trap_tube_env.ACTIONS.left.up
ld = trap_tube_env.ACTIONS.left.down
ll = trap_tube_env.ACTIONS.left.left
lr = trap_tube_env.ACTIONS.left.right
ru = trap_tube_env.ACTIONS.right.up
rd = trap_tube_env.ACTIONS.right.down
rl = trap_tube_env.ACTIONS.right.left
rr = trap_tube_env.ACTIONS.right.right


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
            food_position=self._food_position,
            tool_category=trap_tube_env.TOOL)

    def make_colors(self):
        return trap_tube_env.base_colors


class TrapTubeEnvTest(parameterized.TestCase):

    def setUp(self):
        super(TrapTubeEnvTest, self).setUp()
        self._render = False

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

    def testActionsWithToolNoImpassables(self):
        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                ' a          ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(3 + 1, 1),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, uu, render=self._render)
        self.assertTransition(env, ud, render=self._render)
        self.assertTransition(env, ul, render=self._render)
        self.assertTransition(env, ur, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                ' a          ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(3 + 1, 1),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, du, render=self._render)
        self.assertTransition(env, dd, render=self._render)
        self.assertTransition(env, dl, render=self._render)
        self.assertTransition(env, dr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                ' a  u  n    ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(3 + 1, 2),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, ru, render=self._render)
        self.assertTransition(env, rd, render=self._render)
        self.assertTransition(env, rl, render=self._render)
        self.assertTransition(env, rr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '  a u  n    ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(3 + 1, 1),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, lu, render=self._render)
        self.assertTransition(env, ld, render=self._render)
        self.assertTransition(env, ll, render=self._render)
        self.assertTransition(env, lr, render=self._render)

    def testActionsWithToolTransitionIntoTube(self):
        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '   a        ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(4, 3),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, uu, render=self._render)
        self.assertTransition(env, ud, render=self._render)
        self.assertTransition(env, ul, render=self._render)
        self.assertTransition(env, ur, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '   a        ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(4, 3),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, du, render=self._render)
        self.assertTransition(env, dd, render=self._render)
        self.assertTransition(env, dl, render=self._render)
        self.assertTransition(env, dr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '    a       ',
                '            ',
                '            ',
            ],
            tool_position=(6, 3),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, lu, render=self._render)
        self.assertTransition(env, ld, render=self._render)
        self.assertTransition(env, ll, render=self._render)
        self.assertTransition(env, lr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '   a        ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(5, 4),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, ru, render=self._render)
        self.assertTransition(env, rd, render=self._render)
        self.assertTransition(env, rl, render=self._render)
        self.assertTransition(env, rr, render=self._render)

    def testActionsWithToolImpassableTubeRight(self):
        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '   awwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(3, 3),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, uu, render=self._render)
        self.assertTransition(env, ud, render=self._render)
        self.assertTransition(env, ul, render=self._render)
        self.assertNoTransition(env, ur, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '   au  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(7, 3),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, du, render=self._render)
        self.assertTransition(env, dd, render=self._render)
        self.assertTransition(env, dl, render=self._render)
        self.assertNoTransition(env, dr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '   awwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(4, 2),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, lu, render=self._render)
        self.assertTransition(env, ld, render=self._render)
        self.assertTransition(env, ll, render=self._render)
        self.assertNoTransition(env, lr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '   awwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(4, 4),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, ru, render=self._render)
        self.assertTransition(env, rd, render=self._render)
        self.assertTransition(env, rl, render=self._render)
        self.assertNoTransition(env, rr, render=self._render)

    def testActionsWithToolImpassableTubeLeft(self):
        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwwwa   ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(3, 8),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, uu, render=self._render)
        self.assertTransition(env, ud, render=self._render)
        self.assertNoTransition(env, ul, render=self._render)
        self.assertTransition(env, ur, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  na   ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(7, 8),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, du, render=self._render)
        self.assertTransition(env, dd, render=self._render)
        self.assertNoTransition(env, dl, render=self._render)
        self.assertTransition(env, dr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwwwa   ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(4, 7),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, lu, render=self._render)
        self.assertTransition(env, ld, render=self._render)
        self.assertNoTransition(env, ll, render=self._render)
        self.assertTransition(env, lr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwwwa   ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(4, 9),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, ru, render=self._render)
        self.assertTransition(env, rd, render=self._render)
        self.assertNoTransition(env, rl, render=self._render)
        self.assertTransition(env, rr, render=self._render)

    def testActionsWithToolImpassableTubeUp(self):
        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '       a    ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(4, 7),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertNoTransition(env, uu, render=self._render)
        self.assertTransition(env, ud, render=self._render)
        self.assertTransition(env, ul, render=self._render)
        self.assertTransition(env, ur, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '       a    ',
                '            ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(7, 7),
            tool_size=4,
            tool_direction=0,
            food_position=(4, 4 + 1))
        self.assertNoTransition(env, du, render=self._render)
        self.assertTransition(env, dd, render=self._render)
        self.assertTransition(env, dl, render=self._render)
        self.assertTransition(env, dr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '       a    ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(5, 6),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 3))
        self.assertNoTransition(env, lu, render=self._render)
        self.assertTransition(env, ld, render=self._render)
        self.assertTransition(env, ll, render=self._render)
        self.assertTransition(env, lr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '       a    ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(5, 8),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertNoTransition(env, ru, render=self._render)
        self.assertTransition(env, rd, render=self._render)
        self.assertTransition(env, rl, render=self._render)
        self.assertTransition(env, rr, render=self._render)

    def testActionsWithToolImpassableTubeDown(self):
        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '            ',
                '       a    ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
            ],
            tool_position=(1, 7),
            tool_size=4,
            tool_direction=0,
            food_position=(7, 4 + 1))
        self.assertTransition(env, uu, render=self._render)
        self.assertNoTransition(env, ud, render=self._render)
        self.assertTransition(env, ul, render=self._render)
        self.assertTransition(env, ur, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '       a    ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(4, 7),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, du, render=self._render)
        self.assertNoTransition(env, dd, render=self._render)
        self.assertTransition(env, dl, render=self._render)
        self.assertTransition(env, dr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '       a    ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(3, 6),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 3))
        self.assertTransition(env, lu, render=self._render)
        self.assertNoTransition(env, ld, render=self._render)
        self.assertTransition(env, ll, render=self._render)
        self.assertTransition(env, lr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '       a    ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(3, 8),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, ru, render=self._render)
        self.assertNoTransition(env, rd, render=self._render)
        self.assertTransition(env, rl, render=self._render)
        self.assertTransition(env, rr, render=self._render)

    def testActionsWithToolBoundariesLeft(self):
        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                'a   u  n    ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(1, 0),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, uu, render=self._render)
        self.assertTransition(env, ud, render=self._render)
        self.assertNoTransition(env, ul, render=self._render)
        self.assertTransition(env, ur, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                'a   u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(7, 0),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, du, render=self._render)
        self.assertTransition(env, dd, render=self._render)
        self.assertNoTransition(env, dl, render=self._render)
        self.assertTransition(env, dr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                ' a  u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(6, 0),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, lu, render=self._render)
        self.assertTransition(env, ld, render=self._render)
        self.assertNoTransition(env, ll, render=self._render)
        self.assertTransition(env, lr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                'a   u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(6, 1),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, ru, render=self._render)
        self.assertTransition(env, rd, render=self._render)
        self.assertNoTransition(env, rl, render=self._render)
        self.assertTransition(env, rr, render=self._render)

    def testActionsWithToolBoundariesUp(self):
        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                ' a  mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(0, 1),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertNoTransition(env, uu, render=self._render)
        self.assertTransition(env, ud, render=self._render)
        self.assertTransition(env, ul, render=self._render)
        self.assertTransition(env, ur, render=self._render)

        env = TestEnv(
            art=[
                ' a          ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(1, 1),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertNoTransition(env, du, render=self._render)
        self.assertTransition(env, dd, render=self._render)
        self.assertTransition(env, dl, render=self._render)
        self.assertTransition(env, dr, render=self._render)

        env = TestEnv(
            art=[
                '  a         ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(0, 1),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertNoTransition(env, lu, render=self._render)
        self.assertTransition(env, ld, render=self._render)
        self.assertTransition(env, ll, render=self._render)
        self.assertTransition(env, lr, render=self._render)

        env = TestEnv(
            art=[
                ' a          ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(0, 2),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertNoTransition(env, ru, render=self._render)
        self.assertTransition(env, rd, render=self._render)
        self.assertTransition(env, rl, render=self._render)
        self.assertTransition(env, rr, render=self._render)

    def testActionsWithToolBoundariesDown(self):
        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                ' a          ',
            ],
            tool_position=(7, 1),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, uu, render=self._render)
        self.assertNoTransition(env, ud, render=self._render)
        self.assertTransition(env, ul, render=self._render)
        self.assertTransition(env, ur, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                ' a  wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(8, 1),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, du, render=self._render)
        self.assertNoTransition(env, dd, render=self._render)
        self.assertTransition(env, dl, render=self._render)
        self.assertTransition(env, dr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '  a         ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(8, 1),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, lu, render=self._render)
        self.assertNoTransition(env, ld, render=self._render)
        self.assertTransition(env, ll, render=self._render)
        self.assertTransition(env, lr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                ' a          ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(8, 2),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, ru, render=self._render)
        self.assertNoTransition(env, rd, render=self._render)
        self.assertTransition(env, rl, render=self._render)
        self.assertTransition(env, rr, render=self._render)

    def testActionsWithToolBoundariesRight(self):
        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n   a',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(1, 11),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, uu, render=self._render)
        self.assertTransition(env, ud, render=self._render)
        self.assertTransition(env, ul, render=self._render)
        self.assertNoTransition(env, ur, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n   a',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(6, 11),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, du, render=self._render)
        self.assertTransition(env, dd, render=self._render)
        self.assertTransition(env, dl, render=self._render)
        self.assertNoTransition(env, dr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n   a',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(5, 10),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, lu, render=self._render)
        self.assertTransition(env, ld, render=self._render)
        self.assertTransition(env, ll, render=self._render)
        self.assertNoTransition(env, lr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n  a ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(5, 11),
            tool_size=4,
            tool_direction=0,
            food_position=(4 + 1, 4 + 1))
        self.assertTransition(env, ru, render=self._render)
        self.assertTransition(env, rd, render=self._render)
        self.assertTransition(env, rl, render=self._render)
        self.assertNoTransition(env, rr, render=self._render)

    def testActionsWithToolFoodTrap(self):
        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '       a    ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(5, 6),
            tool_size=4,
            tool_direction=0,
            food_position=(5, 5))
        self.assertNoTransition(env, ll, render=self._render)

    def testActionsWithToolFoodTube(self):
        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '      a     ',
                '            ',
                '            ',
            ],
            tool_position=(6, 5),
            tool_size=4,
            tool_direction=0,
            food_position=(5, 5))
        self.assertNoTransition(env, lu, render=self._render)

    def testActionsTube(self):
        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '       a    ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(0, 0),
            tool_size=4,
            tool_direction=0,
            food_position=(5, 5))
        self.assertNoTransition(env, dd, render=self._render)

    def testActionsExit(self):
        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  na   ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(0, 0),
            tool_size=4,
            tool_direction=0,
            food_position=(5, 5))
        self.assertNoTransition(env, ll, render=self._render)

    def testActionsTrap(self):
        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '   au  n    ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(0, 0),
            tool_size=4,
            tool_direction=0,
            food_position=(5, 5))
        self.assertNoTransition(env, rr, render=self._render)

    def testActionsTool(self):
        env = TestEnv(
            art=[
                ' a          ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '            ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(0, 0),
            tool_size=4,
            tool_direction=0,
            food_position=(5, 5))
        self.assertTransition(env, rr, render=self._render)

    def testActionsWithToolFoodExit(self):
        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '    a       ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(5, 5),
            tool_size=4,
            tool_direction=0,
            food_position=(5, 6))
        self.assertTransition(env, rr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '     a      ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(5, 6),
            tool_size=4,
            tool_direction=0,
            food_position=(5, 7))
        self.assertTransition(env, rr, render=self._render)

        env = TestEnv(
            art=[
                '            ',
                '            ',
                '            ',
                '            ',
                '    mmmm    ',
                '    u  n    ',
                '    u  n    ',
                '    wwww    ',
                '         a  ',
                '            ',
                '            ',
                '            ',
            ],
            tool_position=(5, 8),
            tool_size=4,
            tool_direction=0,
            food_position=(5, 7))
        self.assertTransition(env, ll, render=self._render)


if __name__ == '__main__':
    absltest.main()
