from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym.envs.registration import register
from gym_tool_use.bridge_building import BridgeBuildingEnv


# BridgeBuilding-v{num_boxes - 1}
register(
    id='BridgeBuilding-v0',
    entry_point='gym_tool_use.bridge_building:BridgeBuildingEnv',
    kwargs={'max_steps': 25, 'level': 1})
register(
    id='BridgeBuilding-v1',
    entry_point='gym_tool_use.bridge_building:BridgeBuildingEnv',
    kwargs={'max_steps': 25, 'level': 2})
register(
    id='BridgeBuilding-v2',
    entry_point='gym_tool_use.bridge_building:BridgeBuildingEnv',
    kwargs={'max_steps': 25, 'level': 3})

# Generate random number of boxes.
register(
    id='BridgeBuilding-v3',
    entry_point='gym_tool_use.bridge_building:BridgeBuildingEnv',
    kwargs={'max_steps': 25, 'level': -1})