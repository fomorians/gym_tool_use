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

# Citation

If you use this code in your work, please cite the following:

 ```
@ARTICLE{2019arXiv190702050W,
       author = {{Wenke}, Sam and {Saunders}, Dan and {Qiu}, Mike and {Fleming}, Jim},
        title = "{Reasoning and Generalization in RL: A Tool Use Perspective}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Neural and Evolutionary Computing, Computer Science - Artificial Intelligence, Computer Science - Machine Learning},
         year = "2019",
        month = "Jul",
          eid = {arXiv:1907.02050},
        pages = {arXiv:1907.02050},
archivePrefix = {arXiv},
       eprint = {1907.02050},
 primaryClass = {cs.NE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190702050W},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
