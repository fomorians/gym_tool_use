# Gym Tool-Use

Gym tool-use environments.

<hr/>

```sh
$ git clone https://github.com/fomorians/gym_tool_use.git
$ (cd gym_tool_use; pip install -e .)
```

# Usage

```python
import gym
import gym_tool_use

class TrapEnv(gym_tool_use.BaseTrapTubeEnv):
    
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

        tool_position = (3, 3)
        food_position = (4, 4)
        tool_size = 4
        return gym_tool_use.TrapTubeConfig(
            art=art,
            tool_position=tool_position,
            tool_size=tool_size,
            fake_tool_position=(-1, -1),
            fake_tool_size=tool_size,
            food_position=food_position)

actions = [
    gym_tool_use.ACTIONS.move.right,
    gym_tool_use.ACTIONS.move.right,
    gym_tool_use.ACTIONS.move.right,
    gym_tool_use.ACTIONS.move.up,
    gym_tool_use.ACTIONS.pull.down,
    gym_tool_use.ACTIONS.move.right,
    gym_tool_use.ACTIONS.move.up,
    gym_tool_use.ACTIONS.pull.right,
    gym_tool_use.ACTIONS.pull.right,
    gym_tool_use.ACTIONS.pull.right,
    gym_tool_use.ACTIONS.pull.right,
    gym_tool_use.ACTIONS.move.up,
    gym_tool_use.ACTIONS.move.up,
    gym_tool_use.ACTIONS.move.up,
    gym_tool_use.ACTIONS.move.up]

state = env.reset()
total_reward = 0.
for action in actions:
    state, reward, done, _ = env.step(action)
    total_reward += reward
    if done:
        break
assert total_reward == 1.
```

# Development

Development is started with `pipenv`.

```sh
$ pipenv install
$ pipenv shell
```
