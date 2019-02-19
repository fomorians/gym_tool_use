"""Tool-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from gym_tool_use import utils


class ToolSet(object):
    """Base tool-set class."""

    def paint(self, art, player_position, goal_position, all_positions, np_random=np.random):
        """Paint interactions and return the new `art` and new `all_positions`."""
        pass


class BridgeBuildingToolSet(ToolSet):
    """Bridge building tool-set."""

    def __init__(self, num_interactions):
        super(BridgeBuildingToolSet, self).__init__()
        assert num_interactions <= 4 or num_interactions >= 0
        self.num_interactions = num_interactions

    def paint(self, art, player_position, goal_position, all_positions, np_random=np.random):
        n = len(art)
        box_ids = np_random.choice(
            list(utils.WATER_BOXES), 
            size=self.num_interactions, 
            replace=False)

        # shallow copy positions to remove edges.
        allowed_positions = list(all_positions)
        edge_range = list(range(1, n - 1))
        low_range = [1] * len(edge_range)
        high_range = [n - 2] * len(edge_range)

        edges = list(zip(low_range, edge_range)) + list(zip(high_range, edge_range))
        edges += list(zip(edge_range, low_range)) + list(zip(edge_range, high_range))

        for coordinate in edges:
            if coordinate in allowed_positions:
                allowed_positions.remove(coordinate)

        for box_id in box_ids:
            water_position = allowed_positions.pop()
            box_position = allowed_positions.pop()
            art = utils.paint(art, [water_position, box_position], ['W', box_id])
        return art, all_positions