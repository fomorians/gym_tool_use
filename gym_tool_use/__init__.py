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
    PerceptualStructuralSymbolicTrapTubeEnv,
)

from gym.envs.registration import register

MAX_EPISODE_STEPS = 100


register(
    id="TrapTube-v0",
    entry_point="gym_tool_use.transfers:TrapTubeEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
)
register(
    id="PerceptualTrapTube-v0",
    entry_point="gym_tool_use.transfers:PerceptualTrapTubeEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
)
register(
    id="StructuralTrapTube-v0",
    entry_point="gym_tool_use.transfers:StructuralTrapTubeEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
)
register(
    id="SymbolicTrapTube-v0",
    entry_point="gym_tool_use.transfers:SymbolicTrapTubeEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
)
register(
    id="PerceptualStructuralTrapTube-v0",
    entry_point="gym_tool_use.transfers:PerceptualStructuralTrapTubeEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
)
register(
    id="PerceptualSymbolicTrapTube-v0",
    entry_point="gym_tool_use.transfers:PerceptualSymbolicTrapTubeEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
)
register(
    id="PerceptualStructuralSymbolicTrapTube-v0",
    entry_point="gym_tool_use.transfers:PerceptualStructuralSymbolicTrapTubeEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
)
register(
    id="StructuralSymbolicTrapTube-v0",
    entry_point="gym_tool_use.transfers:StructuralSymbolicTrapTubeEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
)
