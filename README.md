# Gym Tool Use

[`gym`](http://gym.openai.com/docs/) tool use environments.

<hr/>

```sh
$ pip install gym-tool-use
```

# Usage

```python
import gym_tool_use  # import to register gym envs
env = gym.make("TrapTube-v0")
observation = env.reset()
action = env.action_space.sample()
observation_next, reward, done, info = env.step(action)
image = env.render(mode="rgb_array")  # also supports mode="human"
```

# Environments

The following environments are registered:

- `"TrapTube-v0"` (base task)
- `"PerceptualTrapTube-v0"`
- `"StructuralTrapTube-v0"`
- `"SymbolicTrapTube-v0"`
- `"PerceptualSymbolicTrapTube-v0"`
- `"StructuralSymbolicTrapTube-v0"`
- `"PerceptualStructuralTrapTube-v0"`
- `"PerceptualStructuralSymbolicTrapTube-v0"`

# Baselines

Baseline implementations here: https://github.com/fomorians/tool-use

# Development

Development is started with `pipenv`.

```sh
$ pipenv install
$ pipenv shell
```
