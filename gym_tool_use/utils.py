"""Utils for tool use games."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from gym import spaces


WATER_BOXES = '0123'
BOXES = '0123' # + '456789'
COLORS = {
    'G': (255,   0,   0),
    'P': (  0, 255,   0),
    'W': (  0,   0, 255),
    '#': (  0, 255, 255),
}
BOX_COLORS = {box: (255, 255, 0) for box in BOXES}
COLORS.update(BOX_COLORS)

ACTION_SPACE = spaces.Discrete(4 + 1)


def bfs(grid, start, width, height):
    """Breadth-first search.
    https://stackoverflow.com/questions/47896461/get-shortest-path-to-a-cell-in-a-2d-array-python
    """
    wall, clear, goal = '#', ' ', 'G'
    queue = collections.deque([[start]])
    seen = set([start])
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if grid[x][y] == goal:
            return path
        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= x2 < width and 0 <= y2 < height and grid[x2][y2] != wall and (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))


def paint(art, positions, characters):
    """Paint characters into their positions in the art."""
    for position, character in zip(positions, characters):
        art[position[0]][position[1]] = character
    return art