"""Sprites for tool use games."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pycolab.prefab_parts import sprites as prefab_sprites

from gym_tool_use import utils


class BoxSprite(prefab_sprites.MazeWalker):
    """Base box sprite."""

    def __init__(self, corner, position, character):
        impassable = set(utils.BOXES + 'PGB') - set(character)
        super(BoxSprite, self).__init__(
            corner, 
            position, 
            character, 
            impassable,
            confined_to_board=True)

        # Remove the boxes that aren't on the map.
        if position == (0, 0):
            self._teleport((-1, -1))

    def update(self, actions, board, layers, backdrop, things, the_plot):
        """Move boxes around."""
        del backdrop  # Unused.
        rows, cols = self.position

        if actions == 0:    # go upward?
            if (rows + 1) < board.shape[0]:
                if layers['P'][rows + 1, cols]: 
                    prev = self.position
                    self._north(board, the_plot)
                    if self.position != prev:
                        the_plot.info['moved_box'] = 1
                    new_rows, new_cols = self.position
                    if layers['W'][new_rows, new_cols]:
                        the_plot.info['moved_box_into_water'] = 1
            else:
                self._stay(board, the_plot)
        elif actions == 1:  # go downward?
            if (rows - 1) > 0:
                if layers['P'][rows - 1, cols]: 
                    prev = self.position
                    self._south(board, the_plot)
                    if self.position != prev:
                        the_plot.info['moved_box'] = 1
                    new_rows, new_cols = self.position
                    if layers['W'][new_rows, new_cols]:
                        the_plot.info['moved_box_into_water'] = 1
            else:
                self._stay(board, the_plot)
        elif actions == 2:  # go leftward?
            if (cols + 1) < board.shape[1]:
                if layers['P'][rows, cols + 1]:
                    prev = self.position
                    self._west(board, the_plot)
                    if self.position != prev:
                        the_plot.info['moved_box'] = 1
                    new_rows, new_cols = self.position
                    if layers['W'][new_rows, new_cols]:
                        the_plot.info['moved_box_into_water'] = 1
            else:
                self._stay(board, the_plot)
        elif actions == 3:  # go rightward?
            if (cols - 1) > 0:
                if layers['P'][rows, cols - 1]: 
                    prev = self.position
                    self._east(board, the_plot)
                    if self.position != prev:
                        the_plot.info['moved_box'] = 1
                    new_rows, new_cols = self.position
                    if layers['W'][new_rows, new_cols]:
                        the_plot.info['moved_box_into_water'] = 1
            else:
                self._stay(board, the_plot)


class WaterBoxSprite(BoxSprite):
    """Water box."""

    def update(self, actions, board, layers, backdrop, things, the_plot):
        """Fill water spaces to create bridges."""
        rows, cols = self.position

        # Don't move if already in water.
        # Allow the agent to move across me now.
        if layers['W'][rows, cols]:
            self._stay(board, the_plot)
            if 'P' in self._impassable:
                self._impassable.remove('P')
                things['P']._impassable.remove(self.character)
            return

        super(WaterBoxSprite, self).update(
            actions, board, layers, backdrop, things, the_plot)


class PlayerSprite(prefab_sprites.MazeWalker):
    """Base player sprite."""

    def __init__(self, corner, position, character):
        super(PlayerSprite, self).__init__(
            corner, 
            position, 
            character, 
            utils.BOXES + 'W',
            confined_to_board=True)

    def _on_bridge(self, things):
        for box in utils.WATER_BOXES:
            is_on_bridge = np.logical_and(
                things[box].position == (self.position[0], self.position[1]), 
                things['W'].curtain[self.position[0], self.position[1]])
            if is_on_bridge:
                return True
        return False

    def update(self, actions, board, layers, backdrop, things, the_plot):
        """Handles player logic."""
        del backdrop, layers  # Unused.

        was_on_bridge = self._on_bridge(things)
        rows, cols = self.position
        if actions == 0:    # go upward?
            self._north(board, the_plot)
        elif actions == 1:  # go downward?
            self._south(board, the_plot)
        elif actions == 2:  # go leftward?
            self._west(board, the_plot)
        elif actions == 3:  # go rightward?
            self._east(board, the_plot)
        else:
            self._stay(board, the_plot)

        is_on_bridge = self._on_bridge(things)
        if was_on_bridge and not is_on_bridge:
            the_plot.info['moved_across_bridge'] = 1
        elif is_on_bridge:
            the_plot.info['stayed_on_bridge'] = 1