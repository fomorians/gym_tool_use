# Gym Tool-Use

Gym tool-use environments.

<hr/>

```sh
$ git clone https://github.com/fomorians/gym_tool_use.git
$ (cd gym_tool_use; pip install -e .)
```

# Usage

```sh
import gym
import gym_tool_use

train_env = gym.make('ToolUse-v0')
test_env = gym.make('BridgeBuilding-v0')
```

# Development

Development is started with `pipenv`.

```sh
$ pipenv install
$ pipenv shell
```
