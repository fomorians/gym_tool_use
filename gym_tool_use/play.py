"""Play the trap tube environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np

from gym_tool_use import trap_tube_env
from gym_tool_use import transfers


keys_to_actions = {
    'ww': trap_tube_env.ACTIONS.up.up,
    'ws': trap_tube_env.ACTIONS.up.down,
    'wa': trap_tube_env.ACTIONS.up.left,
    'wd': trap_tube_env.ACTIONS.up.right,
    'sw': trap_tube_env.ACTIONS.down.up,
    'ss': trap_tube_env.ACTIONS.down.down,
    'sa': trap_tube_env.ACTIONS.down.left,
    'sd': trap_tube_env.ACTIONS.down.right,
    'aw': trap_tube_env.ACTIONS.left.up,
    'as': trap_tube_env.ACTIONS.left.down,
    'aa': trap_tube_env.ACTIONS.left.left,
    'ad': trap_tube_env.ACTIONS.left.right,
    'dw': trap_tube_env.ACTIONS.right.up,
    'ds': trap_tube_env.ACTIONS.right.down,
    'da': trap_tube_env.ACTIONS.right.left,
    'dd': trap_tube_env.ACTIONS.right.right,
}


def get_action():
    action_str = input('Choose action:')
    assert action_str, '`input` must not be none.'
    assert len(action_str) == 2, '`input` must 2 characters.'
    grasp_id = action_str[0]
    move_id = action_str[1]
    assert grasp_id in ['a', 'w', 's', 'd']
    assert move_id in ['a', 'w', 's', 'd']
    return keys_to_actions[action_str]


def record_episode(env):
    states = []
    actions = []
    rewards = []
    dones = []
    weights = []
    state = env.reset()
    done = False
    env.render()
    while True:
        weights.append(~done)
        states.append(state)
        env.render()
        while True:
            try:
                action = get_action()
                break

            except AssertionError:
                print('Got an error, try again.')

        state, reward, done, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        if done:
            break
    env.close()
    return (
        np.stack(states, axis=0),
        np.stack(actions, axis=0),
        np.stack(rewards, axis=0).astype(np.float32),
        np.stack(dones, axis=0),
        np.stack(weights, axis=0).astype(np.float32))


if __name__ == "__main__":
    import argparse

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
    parser.add_argument(type=str, dest='trial_name')
    parser.add_argument(
        '--seed', default=0, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    transfer = args.transfer
    if transfer == 'p':
        constructor = transfers.PerceptualTrapTubeEnv
    elif transfer == 'st':
        constructor = transfers.StructuralTrapTubeEnv
    elif transfer == 'sy':
        constructor = transfers.SymbolicTrapTubeEnv
    elif transfer == 'stsy':
        constructor = transfers.StructuralSymbolicTrapTubeEnv
    elif transfer == 'pst':
        constructor = transfers.PerceptualStructuralTrapTubeEnv
    elif transfer == 'psy':
        constructor = transfers.PerceptualSymbolicTrapTubeEnv
    elif transfer == 'pstsy':
        constructor = transfers.PerceptualStructuralSymbolicTrapTubeEnv
    elif transfer == 'n':
        constructor = transfers.TrapTubeEnv
    env = constructor()
    env.seed(args.seed)

    try:
        trial_states = []
        trial_actions = []
        trial_rewards = []
        trial_dones = []
        trial_weights = []
        save_episode = False
        while True:
            states, actions, rewards, dones, weights = record_episode(env)
            save_episode = True
            trial_states.append(states)
            trial_actions.append(actions)
            trial_rewards.append(rewards)
            trial_dones.append(dones)
            trial_weights.append(weights)

    except KeyboardInterrupt:
        print('Exiting.')
        if save_episode:
            while True:
                try:
                    np.savez(
                        args.trial_name,
                        states=trial_states,
                        actions=trial_actions,
                        rewards=trial_rewards,
                        dones=trial_dones,
                        weights=trial_weights)
                    break
                except KeyboardInterrupt:
                    print('Failed to save, trying again.')
