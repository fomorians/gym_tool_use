# Gym Tool-Use

Gym tool-use environments.

<hr/>

```sh
$ git clone https://github.com/fomorians/gym_tool_use.git
$ (cd gym_tool_use; pip install -e .)
```

# Environments

## Bridge Building

<p align="center">
    <img src="bridge_building.gif" alt="Bridge Building">
</p>

### `gym.make('BridgeBuilding-v0')`

A game that demonstrates an agent's ability to use tools to build other tools in order to achieve a goal. 

The agent must learn to cross the water to achieve a goal by building a "bridge" using objects (tools) from the environment. There can be multiple bridges and goals.

## Example:

```python
import gym
import gym_tool_use

env = gym.make('BridgeBuilding-v0')
state = env.reset()
total_reward = 0.
env.render()

actions = [3, 3, 3, 3,  # Right
           0, 0, 0, 0,  # Up (build first bridge.)
           1, 1, 1, 1,  # Down
           2, 2, 2,     # Left
           0, 0, 0, 0]  # Up (build second bridge.)

for action in actions:
    _, reward, _, _ = env.step(action)
    total_reward += reward
    env.render()
env.close()
print('Total reward = {}'.format(total_reward))
# >>> 2.
```

## Testing

```sh
$ python -m gym_tool_use.bridge_building_test
```

# Development

Development is started with `pipenv`.

```sh
$ pipenv install
$ pipenv shell
```

