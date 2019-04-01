"""Tool-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class ToolSet(object):
    """Base class for ToolSets."""

    def paint(self, 
              art, 
              what_lies_beneath,
              player_position, 
              goal_position, 
              all_positions, 
              prefill_positions,
              shortest_path=None, 
              np_random=np.random):
        pass