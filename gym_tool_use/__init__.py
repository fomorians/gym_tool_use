from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym_tool_use.trap_tube_env import BaseTrapTubeEnv, TrapTubeConfig, ACTIONS
from gym_tool_use.transfers import (
    PerceptualTrapTubeEnv,
    StructuralTrapTubeEnv,
    SymbolicTrapTubeEnv,
    StructuralSymbolicTrapTubeEnv,
    PerceptualStructuralTrapTubeEnv,
    PerceptualSymbolicTrapTubeEnv,
    PerceptualStructuralSymbolicTrapTubeEnv)

from gym.envs.registration import register


register(
    id='TrapTube-v0',
    entry_point='gym_tool_use.transfers:TrapTubeEnv')
register(
    id='PerceptualTrapTube-v0',
    entry_point='gym_tool_use.transfers:PerceptualTrapTubeEnv')
register(
    id='StructuralTrapTube-v0',
    entry_point='gym_tool_use.transfers:StructuralTrapTubeEnv')
register(
    id='SymbolicTrapTube-v0',
    entry_point=('gym_tool_use.transfers:SymbolicTrapTubeEnv'))
register(
    id='PerceptualStructuralTrapTube-v0',
    entry_point='gym_tool_use.transfers:PerceptualStructuralTrapTubeEnv')
register(
    id='PerceptualSymbolicTrapTube-v0',
    entry_point='gym_tool_use.transfers:PerceptualSymbolicTrapTubeEnv')
register(
    id='PerceptualStructuralSymbolicTrapTube-v0',
    entry_point=(
        'gym_tool_use.transfers:PerceptualStructuralSymbolicTrapTubeEnv'))
register(
    id='StructuralSymbolicTrapTube-v0',
    entry_point=(
        'gym_tool_use.transfers:StructuralSymbolicTrapTubeEnv'))
