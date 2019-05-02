"""Tests for trap tube environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized

from gym_tool_use import trap_tube_env


class TrapEnv(trap_tube_env.BaseTrapTubeEnv):

    successful_actions = [
        trap_tube_env.ACTIONS.move.right,
        trap_tube_env.ACTIONS.move.right,
        trap_tube_env.ACTIONS.move.right,
        trap_tube_env.ACTIONS.move.up,
        trap_tube_env.ACTIONS.pull.down,
        trap_tube_env.ACTIONS.move.right,
        trap_tube_env.ACTIONS.move.up,
        trap_tube_env.ACTIONS.pull.right,
        trap_tube_env.ACTIONS.pull.right,
        trap_tube_env.ACTIONS.pull.right,
        trap_tube_env.ACTIONS.pull.right,
        trap_tube_env.ACTIONS.move.up,
        trap_tube_env.ACTIONS.move.up,
        trap_tube_env.ACTIONS.move.up,
        trap_tube_env.ACTIONS.move.up]

    def _make_trap_tube_config(self):
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
            tool_position=(3, 3),
            tool_size=4,
            tool_direction=0,
            fake_tool_position=(-1, -1),
            fake_tool_size=0,
            fake_tool_direction=0,
            food_position=(4, 4))


class FakeTrapEnv(trap_tube_env.BaseTrapTubeEnv):

    successful_actions = [
        trap_tube_env.ACTIONS.move.right,
        trap_tube_env.ACTIONS.move.right,
        trap_tube_env.ACTIONS.move.right,
        trap_tube_env.ACTIONS.move.up,
        trap_tube_env.ACTIONS.pull.down,
        trap_tube_env.ACTIONS.move.right,
        trap_tube_env.ACTIONS.move.up,
        trap_tube_env.ACTIONS.pull.right,
        trap_tube_env.ACTIONS.pull.right,
        trap_tube_env.ACTIONS.pull.right,
        trap_tube_env.ACTIONS.move.right,
        trap_tube_env.ACTIONS.move.right,
        trap_tube_env.ACTIONS.move.up,
        trap_tube_env.ACTIONS.move.up,
        trap_tube_env.ACTIONS.move.up,
        trap_tube_env.ACTIONS.move.left,
        trap_tube_env.ACTIONS.move.left]

    def _make_trap_tube_config(self):
        art = [
            '          ',
            '          ',
            '          ',
            '  mmmmmm  ',
            '  w    w  ',
            '  w    w  ',
            '  mmmmmm  ',
            '          ',
            'a         ',
            '          ',
        ]
        return trap_tube_env.TrapTubeConfig(
            art=art,
            tool_position=(3, 3),
            tool_size=4,
            tool_direction=0,
            fake_tool_position=(-1, -1),
            fake_tool_size=0,
            fake_tool_direction=0,
            food_position=(4, 4))


class FakeToolEnv(trap_tube_env.BaseTrapTubeEnv):

    successful_actions = [
        trap_tube_env.ACTIONS.move.right,
        trap_tube_env.ACTIONS.move.right,
        trap_tube_env.ACTIONS.move.right,
        trap_tube_env.ACTIONS.move.up,
        trap_tube_env.ACTIONS.pull.down,
        trap_tube_env.ACTIONS.move.right,
        trap_tube_env.ACTIONS.move.right,
        trap_tube_env.ACTIONS.move.right,
        trap_tube_env.ACTIONS.move.right,
        trap_tube_env.ACTIONS.move.right,
        trap_tube_env.ACTIONS.move.up,
        trap_tube_env.ACTIONS.move.up,
        trap_tube_env.ACTIONS.move.up,
        trap_tube_env.ACTIONS.pull.right,
        trap_tube_env.ACTIONS.move.up,
        trap_tube_env.ACTIONS.move.left,
        trap_tube_env.ACTIONS.push.down,
        trap_tube_env.ACTIONS.move.left,
        trap_tube_env.ACTIONS.move.left,
        trap_tube_env.ACTIONS.move.left,
        trap_tube_env.ACTIONS.move.left]

    def _make_trap_tube_config(self):
        art = [
            '          ',
            '          ',
            '          ',
            '  mmmmmm  ',
            '  w       ',
            '  w       ',
            '  mmmmmm  ',
            '          ',
            'a         ',
            '          ',
        ]
        return trap_tube_env.TrapTubeConfig(
            art=art,
            tool_position=(4, 7),
            tool_size=2,
            tool_direction=0,
            fake_tool_position=(3, 3),
            fake_tool_size=4,
            fake_tool_direction=0,
            food_position=(4, 4))


class TrapTubeEnvTest(parameterized.TestCase):

    def rewardForEnvActions(self, env, actions):
        state = env.reset()
        total_reward = 0.
        for action in actions:
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward

    @parameterized.named_parameters(
        ('Trap', TrapEnv, 1., TrapEnv.successful_actions),
        ('FakeTrap', FakeTrapEnv, 1., FakeTrapEnv.successful_actions),
        ('FakeTool', FakeToolEnv, 1., FakeToolEnv.successful_actions))
    def testTotalRewards(self, constructor, expected_total_reward, actions):
        env = constructor()
        env.seed(42)
        total_reward = self.rewardForEnvActions(env, actions)
        env.close()
        self.assertEqual(total_reward, expected_total_reward)


if __name__ == '__main__':
    absltest.main()
