from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym_tool_use import toolsets

from gym_tool_use.envs.bridge_building import BridgeBuildingEnv
from gym_tool_use.envs.tool_use import ToolUseEnv

from gym.envs.registration import register


register(
    id='ToolUse-v0',
    entry_point='gym_tool_use.envs.tool_use:ToolUseEnv',
    kwargs=dict(
        toolsets=[toolsets.BridgeBuildingToolSet(3)], 
        observation_type='layers',
        max_iterations=20))

register(
    id='BridgeBuilding-v0',
    entry_point='gym_tool_use.envs.bridge_building:BridgeBuildingEnv',
    kwargs=dict(
        max_iterations=20, 
        observation_type='layers'))
