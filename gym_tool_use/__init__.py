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
REWARD_THRESHOLD = 1.0


register(
    id="TrapTube-v0",
    entry_point="gym_tool_use.transfers:TrapTubeEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
    reward_threshold=REWARD_THRESHOLD,
)
register(
    id="PerceptualTrapTube-v0",
    entry_point="gym_tool_use.transfers:PerceptualTrapTubeEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
    reward_threshold=REWARD_THRESHOLD,
)
register(
    id="StructuralTrapTube-v0",
    entry_point="gym_tool_use.transfers:StructuralTrapTubeEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
    reward_threshold=REWARD_THRESHOLD,
)
register(
    id="SymbolicTrapTube-v0",
    entry_point="gym_tool_use.transfers:SymbolicTrapTubeEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
    reward_threshold=REWARD_THRESHOLD,
)
register(
    id="PerceptualStructuralTrapTube-v0",
    entry_point="gym_tool_use.transfers:PerceptualStructuralTrapTubeEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
    reward_threshold=REWARD_THRESHOLD,
)
register(
    id="PerceptualSymbolicTrapTube-v0",
    entry_point="gym_tool_use.transfers:PerceptualSymbolicTrapTubeEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
    reward_threshold=REWARD_THRESHOLD,
)
register(
    id="PerceptualStructuralSymbolicTrapTube-v0",
    entry_point="gym_tool_use.transfers:PerceptualStructuralSymbolicTrapTubeEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
    reward_threshold=REWARD_THRESHOLD,
)
register(
    id="StructuralSymbolicTrapTube-v0",
    entry_point="gym_tool_use.transfers:StructuralSymbolicTrapTubeEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
    reward_threshold=REWARD_THRESHOLD,
)
