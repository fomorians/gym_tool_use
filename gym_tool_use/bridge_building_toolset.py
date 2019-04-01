"""Bridge building tool-sets.

This toolset generates water, boxes and bridges.

Water and bridges can exist on the edges but solo-boxes cannot.

An agent or goal will never spawn in the same row as the 'river'.

A 'river' is guarenteed to cover the majority of columns.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from gym_tool_use import utils
from gym_tool_use import toolset


def _count_row_frequency(path, n_rows):
    prev_row = -1
    row_count = [0] * n_rows
    for (row, _) in path:
        row_count[row] += 1
    return row_count


class BridgeBuildingToolSet(toolset.ToolSet):
    """Bridge building tool-set."""

    def paint(self, 
              art, 
              what_lies_beneath,
              player_position, 
              goal_position, 
              all_positions, 
              prefill_positions,
              shortest_path=[], 
              np_random=np.random):
        max_num_boxes = 3
        n = len(art)
        majority_n = 6
        majority_majority_n = 3

        # for each row, count where it contains the shortest path.
        row_count = _count_row_frequency(shortest_path, n)

        # Choose where the 'river' should exist.
        river_rows = list(range(2, 6))  # [2, 5]
        if player_position[0] in river_rows:
            river_rows.remove(player_position[0])
        if goal_position[0] in river_rows:
            river_rows.remove(goal_position[0])
        river_columns = np_random.choice(
            list(range(0, n)), 
            size=majority_n,  # majority
            replace=False)

        # If majority of the river tiles exist in a row 
        # and are part of the shortest path, we need to 
        # move the river. Bridges can only exist as a 
        # minority of the row.
        for row, count in enumerate(row_count):
            if count >= majority_majority_n:
                # Its possible that the agent and goal start 
                # in the same row.
                if row in river_rows:
                    river_rows.remove(row)

        river_row = np_random.choice(river_rows)
        river_positions = []
        for river_column in river_columns:
            river_positions.append((river_row, river_column))

        # Generate the edges of the grid.
        edge_range = list(range(0, n))
        low_range = [0] * len(edge_range)
        high_range = [n - 1] * len(edge_range)
        edges = list(zip(low_range, edge_range)) + list(zip(high_range, edge_range))
        edges += list(zip(edge_range, low_range)) + list(zip(edge_range, high_range))

        allowed_positions = list(all_positions)

        # Boxes can not exist on the edges, 
        # unless it overlaps a water tile.
        for coordinate in edges:
            if coordinate in allowed_positions:
                allowed_positions.remove(coordinate)

        # A box cannot exist in the shortest path, 
        # unless it overlaps a water tile.
        box_positions = []
        for coordinate in shortest_path:
            # There will be at max ((n // 2) + 1) // 2 = 2 positions.
            if coordinate in river_positions:
                box_positions.append(coordinate)

            # Boxes can not exist in the shortest path alone.
            if coordinate in allowed_positions:
                allowed_positions.remove(coordinate)

        # Create boxes.
        # 1, 2, or 3 boxes can exist.
        num_boxes = np_random.choice(
            list(range(max(len(box_positions), 1), max_num_boxes + 1)))
        box_ids = np_random.choice(
            list(utils.WATER_BOXES), 
            size=num_boxes, 
            replace=False).tolist()

        # Render bridges.
        for box_position in box_positions:
            box_id = box_ids.pop()
            art = utils.paint(art, [box_position], [box_id])

        # Render boxes.
        for box_id in box_ids:
            box_position = allowed_positions.pop()
            box_positions.append(box_position)
            art = utils.paint(art, [box_position], [box_id])

        # Render water.
        # Water can exist underneath the box tile as a bridge.
        if 'W' not in prefill_positions:
            prefill_positions['W'] = []
        for river_position in river_positions:
            if river_position in box_positions:
                prefill_positions['W'].append(river_position)
            else:    
                art = utils.paint(art, [river_position], ['W'])

        return art, what_lies_beneath, prefill_positions, all_positions
