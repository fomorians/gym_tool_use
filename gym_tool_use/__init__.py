from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym.envs.registration import register

from gym_tool_use.bridge_building import BridgeBuildingEnv


register(
    id='BridgeBuilding-v0',
    entry_point='gym_tool_use.bridge_building:BridgeBuildingEnv',
    kwargs={'max_steps': 500})