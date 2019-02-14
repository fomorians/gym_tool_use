# Gym Tool-Use

Gym tool-use environments.

<hr/>

```sh
$ git clone https://github.com/fomorians/gym_tool_use.git
$ (cd gym_tool_use; pip install -e .)
```

# Environments

## Bridge Building

Creates a new bridge building game that generates (optionally random): 

+ Positions for the agent `P` to start (i.e. starts on either side of the water), adjacent to the farthest wall from the water `W`.
+ Positions for the goal `G` (i.e. starts on opposite side of `P`).
+ Positions for the boxes. If `v3`, the number of boxes is random from `1:3`. Boxes are positioned 1 step away from the `P` towards the water `W`.
+ Rotations on the game.

for each episode on the environment (reset).

## Testing

TODO(wenkesj).

# Development

Development is started with `pipenv`.

```sh
$ pipenv install
$ pipenv shell
```

