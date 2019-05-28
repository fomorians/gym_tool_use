"""Trap tube examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np

from gym_tool_use import transfers
from gym_tool_use import trap_tube_env


parser = argparse.ArgumentParser()
parser.add_argument(
    choices=[
        'p',
        'st',
        'sy',
        'stsy',
        'pst',
        'psy',
        'pstsy',
        'n',
    ],
    dest='transfer')
args = parser.parse_args()

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

transfer = args.transfer

if transfer == 'p':
    constructor = transfers.PerceptualTrapTubeEnv
    actions = [[uu] * 9 + [ud] + [ul] * 6 + [ud] * 4 + [ll] * 3 + [uu] + [rr] * 3 + [dd] * 2 + [rr] * 4 + [uu, uu]]
    seeds = [43]
elif transfer == 'st':
    constructor = transfers.StructuralTrapTubeEnv
    actions = [[uu] + [ur] * 6 + [rr] + [uu] * 3]
    seeds = [43]
elif transfer == 'sy':
    constructor = transfers.SymbolicTrapTubeEnv
    actions = [
        [uu] + [ur] + [uu] + [rr] * 3 + [ru] * 2,
        [uu, rr] + [uu] * 3 + [rr, rl, uu, rr, dd] + [rr] * 2]
    seeds = [43, 42]
elif transfer == 'stsy':
    constructor = transfers.StructuralSymbolicTrapTubeEnv
    actions = [[uu] + [ur] + [uu] + [rr] * 3 + [ru] * 2]
    seeds = [43]
elif transfer == 'pst':
    constructor = transfers.PerceptualStructuralTrapTubeEnv
    actions = [[uu] * 9 + [ud] + [ul] * 6 + [ud] * 4 + [ll] * 3 + [uu] + [rr] * 3 + [dd] * 2 + [rr] * 4 + [uu, uu]]
    seeds = [43]
elif transfer == 'psy':
    constructor = transfers.PerceptualSymbolicTrapTubeEnv
    actions = [[uu] * 7 + [ll] * 2 + [lr] * 2 + [uu] + [ll] * 4 + [dd] * 3]
    seeds = [43]
elif transfer == 'pstsy':
    constructor = transfers.PerceptualStructuralSymbolicTrapTubeEnv
    actions = [[uu] * 7 + [ll] * 2 + [lr] * 2 + [uu] + [ll] * 4 + [dd] * 3]
    seeds = [43]
elif transfer == 'n':
    constructor = transfers.TrapTubeEnv
    actions = [[uu] + [ur] * 6 + [rr] + [uu] * 3]
    seeds = [43]


for i, seed in enumerate(seeds):
    actions_ = actions[i]
    env = constructor()
    env.seed(seed)
    state = env.reset()
    env.render()
    total_reward = 0.
    for action in actions_:
        state, reward, done, _ = env.step(action)
        env.render()
        total_reward += reward
        if done:
            break
    env.close()
    print(total_reward)
