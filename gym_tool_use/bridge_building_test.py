from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from gym_tool_use import bridge_building


class TestBridgeBuilding(unittest.TestCase):

    def _expect_reward_from_actions_on_level(self, expected_reward, actions, level=0):
        """Simulate the level for 25 steps using the actions and expect the reward."""
        env = bridge_building.BridgeBuildingEnv(
            level=level,
            max_steps=25,
            default_reward=0.)
        state = env.reset()
        env.render()
        total_reward = 0.
        for action in actions:
            _, reward, _, _ = env.step(action)
            total_reward += reward
            env.render()
        env.close()
        self.assertEqual(total_reward, expected_reward)

    def test_expected_rewards(self):
        self._expect_reward_from_actions_on_level(
            1., [3, 3, 3, 3, 0, 0, 0, 0],level=0)
        self._expect_reward_from_actions_on_level(
            -1., [3, 3, 3, 0, 0], level=0)
        self._expect_reward_from_actions_on_level(
            1., [3, 3, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 0, 0], level=0)
        self._expect_reward_from_actions_on_level(
            1., [1, 1, 1, 3, 3, 3, 3, 3, 3], level=1)
        self._expect_reward_from_actions_on_level(
            -1., [1, 1, 3, 3, 3], level=1)
        self._expect_reward_from_actions_on_level(
            0., [3, 3, 3, 3, 0, 0], level=2)
        self._expect_reward_from_actions_on_level(
            0., [1, 1, 1, 3, 3], level=3)
        self._expect_reward_from_actions_on_level(
            2., [1, 1, 1, 3, 3, 3, 3, 3, 3, 1], level=4)


if __name__ == '__main__':
    unittest.main()
