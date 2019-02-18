"""Sprites for tool use games."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pycolab.prefab_parts import sprites as prefab_sprites

from gym_tool_use import utils


class BoxSprite(prefab_sprites.MazeWalker):
    """Base box sprite."""

    def __init__(self, corner, position, character, extra_impassibles='W'):
        impassable = set(utils.BOXES + '#PGBX' + extra_impassibles) - set(character)
        super(BoxSprite, self).__init__(
            corner, 
            position, 
            character, 
            impassable)

        # Remove the boxes that aren't on the map.
        if position == (0, 0):
            self._teleport((-1, -1))

    def update(self, actions, board, layers, backdrop, things, the_plot):
        """Move boxes around."""
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

    def __init__(self, corner, position, character):
        super(WaterBoxSprite, self).__init__(
            corner, 
            position, 
            character, 
            extra_impassibles='')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        """Fill water spaces to create bridges."""
        rows, cols = self.position

        # Don't move if already in water.
        if layers['W'][rows, cols]:
            self._stay(board, the_plot)
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
            utils.BOXES + '#XW')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        """Handles player logic."""
        del backdrop  # Unused.

        rows, cols = self.position
        if actions == 0:    # go upward?
            if (rows - 1) > 0:
                for box in utils.WATER_BOXES:
                    box_is_north = things[box].position == (rows - 1, cols)
                    water_is_north = things['W'].curtain[rows - 1, cols]
                    can_cross_bridge = not layers['#'][rows - 2, cols] if (rows - 2) > 0 else False
                    if box_is_north and water_is_north:
                        if can_cross_bridge:  # cross the bridge?
                            self._teleport((self.virtual_position[0] - 2, 
                                            self.virtual_position[1] + 0))
                            the_plot['moved_across_bridge'] = 1
                        else:
                            self._stay(board, the_plot)
                        return
            self._north(board, the_plot)
        elif actions == 1:  # go downward?
            if (rows + 1) < board.shape[0]:
                for box in utils.WATER_BOXES:
                    box_is_south = things[box].position == (rows + 1, cols)
                    water_is_south = things['W'].curtain[rows + 1, cols]
                    can_cross_bridge = not layers['#'][rows + 2, cols] if (rows + 2) < board.shape[0] else False
                    if box_is_south and water_is_south:
                        if can_cross_bridge:  # cross the bridge?
                            self._teleport((self.virtual_position[0] + 2, 
                                            self.virtual_position[1] + 0))
                            the_plot['moved_across_bridge'] = 1
                        else:
                            self._stay(board, the_plot)
                        return
            self._south(board, the_plot)
        elif actions == 2:  # go leftward?
            if (cols - 1) > 0:
                for box in utils.WATER_BOXES:
                    box_is_west = things[box].position == (rows, cols - 1)
                    water_is_west = things['W'].curtain[rows, cols - 1]
                    can_cross_bridge = not layers['#'][rows, cols - 2] if (cols - 2) > 0 else False
                    if box_is_west and water_is_west:  
                        if can_cross_bridge:  # cross the bridge?
                            self._teleport((self.virtual_position[0] + 0, 
                                            self.virtual_position[1] - 2))
                            the_plot['moved_across_bridge'] = 1
                        else:
                            self._stay(board, the_plot)
                        return
            self._west(board, the_plot)
        elif actions == 3:  # go rightward?
            if (cols + 1) < board.shape[1]:
                for box in utils.WATER_BOXES:
                    box_is_east = things[box].position == (rows, cols + 1)
                    water_is_east = things['W'].curtain[rows, cols + 1]
                    can_cross_bridge = not layers['#'][rows, cols + 2] if (cols + 2) < board.shape[1] else False
                    if box_is_east and water_is_east:  
                        if can_cross_bridge:  # cross the bridge?
                            self._teleport((self.virtual_position[0] + 0, 
                                            self.virtual_position[1] + 2))
                            the_plot['moved_across_bridge'] = 1
                        else:
                            self._stay(board, the_plot)
                        return
            self._east(board, the_plot)
        else:
            self._stay(board, the_plot)
