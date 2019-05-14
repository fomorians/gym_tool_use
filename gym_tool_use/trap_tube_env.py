"""Sprites & Things for tool use games."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import collections

import numpy as np

from gym import spaces

import gym_pycolab

from pycolab import things as plab_things
from pycolab import ascii_art
from pycolab.prefab_parts import sprites as prefab_sprites


TOOL = 'p'
AGENT = 'a'
TUBE1 = 'm'
TUBE2 = 'w'
TRAP = 'u'
EXIT = 'n'
FOOD = 'f'
TASK = 'j'
GROUND = ' '
SYMBOLIC_OBJECTS = [TOOL, TUBE1, TUBE2, TRAP, EXIT]
TOOL_COLOR = (152, 208, 57)
AGENT_COLOR = (113, 57, 208)
TUBE1_COLOR = (57, 152, 208)
TUBE2_COLOR = (57, 152, 208)
TRAP_COLOR = (208, 113, 57)
EXIT_COLOR = (208, 189, 57)
FOOD_COLOR = (208, 57, 77)
GROUND_COLOR = (72, 65, 17)
REWARD = 1.0

Actions = collections.namedtuple(
    'Actions', ['move', 'push', 'pull'])
Directions = collections.namedtuple(
    'Directions', ['up', 'down', 'left', 'right'])

ACTIONS = Actions(
    move=Directions(
        up=[0, 0],
        down=[0, 1],
        left=[0, 2],
        right=[0, 3]),
    push=Directions(
        up=[1, 0],
        down=[1, 1],
        left=[1, 2],
        right=[1, 3]),
    pull=Directions(
        up=[2, 0],
        down=[2, 1],
        left=[2, 2],
        right=[2, 3]))


def actions_equal(actual_action, expected_action):
    return np.all(actual_action == expected_action)


class Tube1Drape(plab_things.Drape):
    """Handles tube logic."""

    def update(self, actions, board, layers, backdrop, things, the_plot):
        pass


class Tube2Drape(plab_things.Drape):
    """Handles tube logic."""

    def update(self, actions, board, layers, backdrop, things, the_plot):
        pass


class TrapDrape(plab_things.Drape):
    """Handles trap logic."""

    def update(self, actions, board, layers, backdrop, things, the_plot):
        pass


class ExitDrape(plab_things.Drape):
    """Handles exit logic."""

    def update(self, actions, board, layers, backdrop, things, the_plot):
        pass


class FoodDrape(plab_things.Drape):

    def __init__(self, curtain, character, position):
        curtain[position] = True
        self._row, self._col = position
        self.has_moved = False
        super(FoodDrape, self).__init__(curtain, character)

    @property
    def position(self):
        return (self._row, self._col)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        """Handles food logic."""
        agent = things[AGENT]
        tool = things[TOOL]
        trap = things[TRAP]
        tube1 = things[TUBE1]
        tube2 = things[TUBE2]

        if actions is None:
            self.has_moved = False
            return

        agent_row, agent_col = agent.position
        food_row, food_col = self.position

        if actions_equal(actions, ACTIONS.push.up):
            agent_is_in_col_range_of_tool = tool.is_in_col_range_of_tool(
                agent_row, agent_col)
            agent_is_south_of_tool = tool.is_south_of_tool(
                agent_row, agent_col)
            agent_can_move_north = agent.can_move_north(board, things)
            food_is_in_col_range_of_tool = tool.is_in_col_range_of_tool(
                food_row, food_col)
            food_is_north_of_tool = tool.is_north_of_tool(food_row, food_col)
            if agent_is_south_of_tool and agent_can_move_north and \
                    agent_is_in_col_range_of_tool and food_is_north_of_tool \
                    and food_is_in_col_range_of_tool:
                if (food_row - 1) < 0:
                    self.has_moved = False
                else:
                    trap_is_north_of_food = trap.curtain[
                        food_row - 1, food_col]
                    tube1_is_north_of_food = tube1.curtain[
                        food_row - 1, food_col]
                    tube2_is_north_of_food = tube2.curtain[
                        food_row - 1, food_col]
                    if trap_is_north_of_food or tube1_is_north_of_food or \
                            tube2_is_north_of_food or (food_row == 0):
                        self.has_moved = False
                    else:
                        self._north(board, things, the_plot)
            else:
                self.has_moved = False

        if actions_equal(actions, ACTIONS.pull.up):
            agent_is_in_col_range_of_tool = tool.is_in_col_range_of_tool(
                agent_row, agent_col)
            agent_is_north_of_tool = tool.is_north_of_tool(
                agent_row, agent_col)
            agent_can_move_north = agent.can_move_north(board, things)
            food_is_in_col_range_of_tool = tool.is_in_col_range_of_tool(
                food_row, food_col)
            food_is_north_of_tool = tool.is_north_of_tool(food_row, food_col)
            if agent_is_north_of_tool and agent_can_move_north and \
                    agent_is_in_col_range_of_tool and food_is_north_of_tool \
                    and food_is_in_col_range_of_tool:
                if (food_row - 1) < 0:
                    self.has_moved = False
                else:
                    trap_is_north_of_food = trap.curtain[
                        food_row - 1, food_col]
                    tube1_is_north_of_food = tube1.curtain[
                        food_row - 1, food_col]
                    tube2_is_north_of_food = tube1.curtain[
                        food_row - 1, food_col]
                    if trap_is_north_of_food or tube1_is_north_of_food or \
                            tube2_is_north_of_food or (food_row == 0):
                        self.has_moved = False
                    else:
                        self._north(board, things, the_plot)
            else:
                self.has_moved = False

        elif actions_equal(actions, ACTIONS.push.down):
            agent_is_in_col_range_of_tool = tool.is_in_col_range_of_tool(
                agent_row, agent_col)
            agent_is_north_of_tool = tool.is_north_of_tool(
                agent_row, agent_col)
            agent_can_move_south = agent.can_move_south(board, things)
            food_is_in_col_range_of_tool = tool.is_in_col_range_of_tool(
                food_row, food_col)
            food_is_south_of_tool = tool.is_south_of_tool(food_row, food_col)
            if agent_is_north_of_tool and agent_can_move_south and \
                    agent_is_in_col_range_of_tool and food_is_south_of_tool \
                    and food_is_in_col_range_of_tool:
                if (food_row + 1) >= board.shape[0]:
                    self.has_moved = False
                else:
                    trap_is_south_of_food = trap.curtain[
                        food_row + 1, food_col]
                    tube1_is_south_of_food = tube1.curtain[
                        food_row + 1, food_col]
                    tube2_is_south_of_food = tube2.curtain[
                        food_row + 1, food_col]
                    if trap_is_south_of_food or tube1_is_south_of_food or \
                            tube2_is_south_of_food:
                        self.has_moved = False
                    else:
                        self._south(board, things, the_plot)
            else:
                self.has_moved = False

        elif actions_equal(actions, ACTIONS.pull.down):
            agent_is_in_col_range_of_tool = tool.is_in_col_range_of_tool(
                agent_row, agent_col)
            agent_is_south_of_tool = tool.is_south_of_tool(
                agent_row, agent_col)
            agent_can_move_south = agent.can_move_south(board, things)
            food_is_in_col_range_of_tool = tool.is_in_col_range_of_tool(
                food_row, food_col)
            food_is_south_of_tool = tool.is_south_of_tool(food_row, food_col)
            if agent_is_south_of_tool and agent_can_move_south and \
                    agent_is_in_col_range_of_tool and food_is_south_of_tool \
                    and food_is_in_col_range_of_tool:
                if (food_row + 1) >= board.shape[0]:
                    self.has_moved = False
                else:
                    trap_is_south_of_food = trap.curtain[
                        food_row + 1, food_col]
                    tube1_is_south_of_food = tube1.curtain[
                        food_row + 1, food_col]
                    tube2_is_south_of_food = tube2.curtain[
                        food_row + 1, food_col]
                    if trap_is_south_of_food or tube1_is_south_of_food or \
                            tube2_is_south_of_food:
                        self.has_moved = False
                    else:
                        self._south(board, things, the_plot)
            else:
                self.has_moved = False

        elif actions_equal(actions, ACTIONS.push.right):
            agent_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(
                agent_row, agent_col)
            agent_is_west_of_tool = tool.is_west_of_tool(agent_row, agent_col)
            agent_can_move_east = agent.can_move_east(board, things)
            food_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(
                food_row, food_col)
            food_is_east_of_tool = tool.is_east_of_tool(food_row, food_col)
            if agent_is_west_of_tool and agent_can_move_east and \
                    agent_is_in_row_range_of_tool and food_is_east_of_tool \
                    and food_is_in_row_range_of_tool:
                if (food_col + 1) >= board.shape[1]:
                    self.has_moved = False
                else:
                    trap_is_east_of_food = trap.curtain[food_row, food_col + 1]
                    tube1_is_east_of_food = tube1.curtain[
                        food_row, food_col + 1]
                    tube2_is_east_of_food = tube2.curtain[
                        food_row, food_col + 1]
                    if trap_is_east_of_food or tube1_is_east_of_food or \
                            tube2_is_east_of_food:
                        self.has_moved = False
                    else:
                        self._east(board, things, the_plot)
            else:
                self.has_moved = False

        elif actions_equal(actions, ACTIONS.pull.left):
            agent_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(
                agent_row, agent_col)
            agent_is_west_of_tool = tool.is_west_of_tool(agent_row, agent_col)
            agent_can_move_west = agent.can_move_west(board, things)
            food_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(
                food_row, food_col)
            food_is_west_of_tool = tool.is_west_of_tool(food_row, food_col)
            if agent_is_west_of_tool and agent_can_move_west and \
                    agent_is_in_row_range_of_tool and food_is_west_of_tool \
                    and food_is_in_row_range_of_tool:
                if (food_col - 1) < 0:
                    self.has_moved = False
                else:
                    trap_is_west_of_food = trap.curtain[food_row, food_col - 1]
                    tube1_is_west_of_food = tube1.curtain[
                        food_row, food_col - 1]
                    tube2_is_west_of_food = tube2.curtain[
                        food_row, food_col - 1]
                    if trap_is_west_of_food or tube1_is_west_of_food or \
                            tube2_is_west_of_food:
                        self.has_moved = False
                    else:
                        self._west(board, things, the_plot)
            else:
                self.has_moved = False

        elif actions_equal(actions, ACTIONS.push.left):
            agent_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(
                agent_row, agent_col)
            agent_is_east_of_tool = tool.is_east_of_tool(agent_row, agent_col)
            agent_can_move_west = agent.can_move_west(board, things)
            food_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(
                food_row, food_col)
            food_is_west_of_tool = tool.is_west_of_tool(food_row, food_col)
            if agent_is_east_of_tool and agent_can_move_west and \
                    agent_is_in_row_range_of_tool and food_is_west_of_tool \
                    and food_is_in_row_range_of_tool:
                if (food_col - 1) < 0:
                    self.has_moved = False
                else:
                    trap_is_west_of_food = trap.curtain[food_row, food_col - 1]
                    tube1_is_west_of_food = tube1.curtain[
                        food_row, food_col - 1]
                    tube2_is_west_of_food = tube2.curtain[
                        food_row, food_col - 1]
                    if trap_is_west_of_food or tube1_is_west_of_food or \
                            tube2_is_west_of_food:
                        self.has_moved = False
                    else:
                        self._west(board, things, the_plot)
            else:
                self.has_moved = False

        elif actions_equal(actions, ACTIONS.pull.right):
            agent_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(
                agent_row, agent_col)
            agent_is_east_of_tool = tool.is_east_of_tool(agent_row, agent_col)
            agent_can_move_east = agent.can_move_east(board, things)
            food_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(
                food_row, food_col)
            food_is_east_of_tool = tool.is_east_of_tool(food_row, food_col)
            if agent_is_east_of_tool and agent_can_move_east and \
                    agent_is_in_row_range_of_tool and food_is_east_of_tool \
                    and food_is_in_row_range_of_tool:
                if (food_col + 1) >= board.shape[1]:
                    self.has_moved = False
                else:
                    trap_is_east_of_food = trap.curtain[food_row, food_col + 1]
                    tube1_is_east_of_food = tube1.curtain[
                        food_row, food_col + 1]
                    tube2_is_east_of_food = tube2.curtain[
                        food_row, food_col + 1]
                    if trap_is_east_of_food or tube1_is_east_of_food or \
                            tube2_is_east_of_food:
                        self.has_moved = False
                    else:
                        self._east(board, things, the_plot)
            else:
                self.has_moved = False

        else:
            self._stay(board, things, the_plot)

    def _north(self, board, things, the_plot):
        if self._row > 0:
            the_plot.info['move_food_north'] = True
            self.curtain[self.position] = False
            self._row -= 1
            self.curtain[self.position] = True
            self.has_moved = True
        else:
            self.has_moved = False

    def _south(self, board, things, the_plot):
        if self._row < board.shape[0]:
            the_plot.info['move_food_south'] = True
            self.curtain[self.position] = False
            self._row += 1
            self.curtain[self.position] = True
            self.has_moved = True
        else:
            self.has_moved = False

    def _east(self, board, things, the_plot):
        if self._col < board.shape[1]:
            the_plot.info['move_food_east'] = True
            self.curtain[self.position] = False
            self._col += 1
            self.curtain[self.position] = True
            self.has_moved = True
        else:
            self.has_moved = False

    def _west(self, board, things, the_plot):
        if self._col > 0:
            the_plot.info['move_food_west'] = True
            self.curtain[self.position] = False
            self._col -= 1
            self.curtain[self.position] = True
            self.has_moved = True
        else:
            self.has_moved = False

    def _stay(self, board, things, the_plot):
        self.has_moved = False


class ToolDrape(plab_things.Drape):
    """Handles tool logic."""

    def __init__(self,
                 curtain,
                 character,
                 position,
                 tool_size=4,
                 tool_direction=0):
        w, h = curtain.shape[0], curtain.shape[1]
        self._row, self._col = position
        self.has_moved = False

        assert tool_direction in [0, 1], '`tool_direction` must be 0 or 1.'
        assert tool_size >= 0, '`tool_size` must be >= 0.'

        self._tool_direction = tool_direction
        self._tool_size = tool_size

        if position[0] != -1 and position[1] != -1:
            assert (position[0] >= 0) and (position[1] >= 0), '`position` < 0'
            # tranpose dimensions according to the tool direction
            d0, d1, p0, p1, s0, s1 = (
                (w, h, position[0], position[1],
                 [row for row in range(
                     self._row, self._row + self._tool_size)],
                 [self._col] * self._tool_size),
                (h, w, position[1], position[0],
                 [self._row] * self._tool_size,
                 [col for col in range(
                     self._col, self._col + self._tool_size)],))[
                         self._tool_direction]
            assert ((p0 + (self._tool_size - 1)) < d0)
            assert (p1 < d1)

            for position in zip(s0, s1):
                curtain[position] = True

        super(ToolDrape, self).__init__(curtain, character)

    def _north(self, board, things, the_plot):
        if self._row == 0:
            self.has_moved = False
            return

        food = things[FOOD]
        row, col = food.position
        food_is_in_col_range_of_tool = self.is_in_col_range_of_tool(row, col)
        food_is_north_of_tool = self.is_north_of_tool(row, col)
        food_has_moved = food.has_moved
        if food_is_north_of_tool and food_is_in_col_range_of_tool and not \
                food_has_moved:
            self.has_moved = False
            return

        if not things[AGENT].can_move_north(board, things):
            self.has_moved = False
            return

        the_plot.info['move_tool_north'] = True
        curtain = np.roll(self.curtain, -1, axis=0)
        np.copyto(self.curtain, curtain)
        self._row -= 1
        self.has_moved = True

    def _south(self, board, things, the_plot):
        if self._row == (board.shape[0] - 1 - (self._tool_size - 1) * int(
                self._tool_direction == 0)):
            self.has_moved = False
            return

        food = things[FOOD]
        row, col = food.position
        food_is_in_col_range_of_tool = self.is_in_col_range_of_tool(row, col)
        food_is_south_of_tool = self.is_south_of_tool(row, col)
        food_has_moved = food.has_moved
        if food_is_south_of_tool and food_is_in_col_range_of_tool and not \
                food_has_moved:
            self.has_moved = False
            return

        if not things[AGENT].can_move_south(board, things):
            self.has_moved = False
            return

        the_plot.info['move_tool_south'] = True
        curtain = np.roll(self.curtain, 1, axis=0)
        np.copyto(self.curtain, curtain)
        self._row += 1
        self.has_moved = True

    def _east(self, board, things, the_plot):
        if self._col == (board.shape[1] - 1 - (self._tool_size - 1) * int(
                self._tool_direction == 1)):
            self.has_moved = False
            return

        food = things[FOOD]
        row, col = food.position
        food_is_in_row_range_of_tool = self.is_in_row_range_of_tool(row, col)
        food_is_east_of_tool = self.is_east_of_tool(row, col)
        food_has_moved = food.has_moved
        if food_is_east_of_tool and food_is_in_row_range_of_tool and not \
                food_has_moved:
            self.has_moved = False
            return

        if not things[AGENT].can_move_east(board, things):
            self.has_moved = False
            return

        curtain = np.roll(self.curtain, 1, axis=1)
        np.copyto(self.curtain, curtain)
        self._col += 1
        self.has_moved = True
        the_plot.info['move_tool_east'] = True

    def _west(self, board, things, the_plot):
        if self._col == 0:
            self.has_moved = False
            return

        food = things[FOOD]
        row, col = food.position
        food_is_in_row_range_of_tool = self.is_in_row_range_of_tool(row, col)
        food_is_west_of_tool = self.is_west_of_tool(row, col)
        food_has_moved = food.has_moved
        if food_is_west_of_tool and food_is_in_row_range_of_tool and not \
                food_has_moved:
            self.has_moved = False
            return

        if not things[AGENT].can_move_west(board, things):
            self.has_moved = False
            return

        the_plot.info['move_tool_west'] = True
        curtain = np.roll(self.curtain, -1, axis=1)
        np.copyto(self.curtain, curtain)
        self._col -= 1
        self.has_moved = True

    def _stay(self, board, things, the_plot):
        self.has_moved = False

    def is_south_of_tool(self, row, col):
        row_offset = self._tool_size if self._tool_direction == 0 else 1
        return row == (self._row + row_offset)

    def is_north_of_tool(self, row, col):
        return row == (self._row - 1)

    def is_west_of_tool(self, row, col):
        return col == (self._col - 1)

    def is_east_of_tool(self, row, col):
        col_offset = self._tool_size if self._tool_direction == 1 else 1
        return col == (self._col + col_offset)

    def _is_in_range_of_tool(self, row, col, direction):
        in_tool_direction = direction == self._tool_direction
        tool_size = (self._tool_size - 1) if in_tool_direction else 0
        direction_size, index = ((self._row, row), (self._col, col))[direction]
        return direction_size <= index <= (direction_size + tool_size)

    def is_in_row_range_of_tool(self, row, col):
        return self._is_in_range_of_tool(row, col, 0)

    def is_in_col_range_of_tool(self, row, col):
        return self._is_in_range_of_tool(row, col, 1)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        agent = things[AGENT]

        if actions is None:
            self.has_moved = False
            return

        row, col = agent.position
        if actions_equal(actions, ACTIONS.push.up):
            is_in_col_range_of_tool = self.is_in_col_range_of_tool(row, col)
            is_south_of_tool = self.is_south_of_tool(row, col)
            if is_in_col_range_of_tool and is_south_of_tool:
                self._north(board, things, the_plot)
            else:
                self.has_moved = False
        elif actions_equal(actions, ACTIONS.pull.down):
            is_in_col_range_of_tool = self.is_in_col_range_of_tool(row, col)
            is_south_of_tool = self.is_south_of_tool(row, col)
            if is_in_col_range_of_tool and is_south_of_tool:
                self._south(board, things, the_plot)
            else:
                self.has_moved = False
        elif actions_equal(actions, ACTIONS.pull.up):
            is_in_col_range_of_tool = self.is_in_col_range_of_tool(row, col)
            is_north_of_tool = self.is_north_of_tool(row, col)
            if is_in_col_range_of_tool and is_north_of_tool:
                self._north(board, things, the_plot)
            else:
                self.has_moved = False
        elif actions_equal(actions, ACTIONS.push.down):
            is_in_col_range_of_tool = self.is_in_col_range_of_tool(row, col)
            is_north_of_tool = self.is_north_of_tool(row, col)
            if is_in_col_range_of_tool and is_north_of_tool:
                self._south(board, things, the_plot)
            else:
                self.has_moved = False
        elif actions_equal(actions, ACTIONS.push.right):
            is_in_row_range_of_tool = self.is_in_row_range_of_tool(row, col)
            is_west_of_tool = self.is_west_of_tool(row, col)
            if is_in_row_range_of_tool and is_west_of_tool:
                self._east(board, things, the_plot)
            else:
                self.has_moved = False
        elif actions_equal(actions, ACTIONS.pull.left):
            is_in_row_range_of_tool = self.is_in_row_range_of_tool(row, col)
            is_west_of_tool = self.is_west_of_tool(row, col)
            if is_in_row_range_of_tool and is_west_of_tool:
                self._west(board, things, the_plot)
            else:
                self.has_moved = False
        elif actions_equal(actions, ACTIONS.push.left):
            is_in_row_range_of_tool = self.is_in_row_range_of_tool(row, col)
            is_east_of_tool = self.is_east_of_tool(row, col)
            if is_in_row_range_of_tool and is_east_of_tool:
                self._west(board, things, the_plot)
            else:
                self.has_moved = False
        elif actions_equal(actions, ACTIONS.pull.right):
            is_in_row_range_of_tool = self.is_in_row_range_of_tool(row, col)
            is_east_of_tool = self.is_east_of_tool(row, col)
            if is_in_row_range_of_tool and is_east_of_tool:
                self._east(board, things, the_plot)
            else:
                self.has_moved = False
        else:
            self._stay(board, things, the_plot)


class AgentSprite(prefab_sprites.MazeWalker):
    """Handles agent logic."""

    def __init__(self, corner, position, character, impassible):
        super(AgentSprite, self).__init__(
            corner,
            position,
            character,
            impassible,
            confined_to_board=True)

    def can_move_south(self, board, things):
        row, col = self.position
        if row == (board.shape[0] - 1):
            return False
        impassables = self.impassable
        for impassable in impassables:
            if impassable not in [TOOL]:
                if things[impassable].curtain[row + 1, col]:
                    return False
        return True

    def can_move_north(self, board, things):
        row, col = self.position
        if row == 0:
            return False
        impassables = self.impassable
        for impassable in impassables:
            if impassable not in [TOOL]:
                if things[impassable].curtain[row - 1, col]:
                    return False
        return True

    def can_move_west(self, board, things):
        row, col = self.position
        if col == 0:
            return False
        impassables = self.impassable
        for impassable in impassables:
            if impassable not in [TOOL]:
                if things[impassable].curtain[row, col - 1]:
                    return False
        return True

    def can_move_east(self, board, things):
        row, col = self.position
        if col == (board.shape[1] - 1):
            return False
        impassables = self.impassable
        for impassable in impassables:
            if impassable not in [TOOL]:
                if things[impassable].curtain[row, col + 1]:
                    return False
        return True

    def update(self, actions, board, layers, backdrop, things, the_plot):
        tool = things[TOOL]
        rows, cols = self.position

        if actions is None:
            return

        if actions_equal(actions, ACTIONS.move.up):
            self._north(board, the_plot)
        elif actions_equal(actions, ACTIONS.push.up):
            self._north(board, the_plot)
        elif actions_equal(actions, ACTIONS.pull.up):
            self._north(board, the_plot)
        elif actions_equal(actions, ACTIONS.move.down):
            self._south(board, the_plot)
        elif actions_equal(actions, ACTIONS.push.down):
            self._south(board, the_plot)
        elif actions_equal(actions, ACTIONS.pull.down):
            self._south(board, the_plot)
        elif actions_equal(actions, ACTIONS.move.left):
            self._west(board, the_plot)
        elif actions_equal(actions, ACTIONS.push.left):
            self._west(board, the_plot)
        elif actions_equal(actions, ACTIONS.pull.left):
            self._west(board, the_plot)
        elif actions_equal(actions, ACTIONS.move.right):
            self._east(board, the_plot)
        elif actions_equal(actions, ACTIONS.push.right):
            self._east(board, the_plot)
        elif actions_equal(actions, ACTIONS.pull.right):
            self._east(board, the_plot)
        else:
            self._stay(board, the_plot)


class TaskDrape(plab_things.Drape):
    """Handles task logic."""

    def update(self, actions, board, layers, backdrop, things, the_plot):
        agent = things[AGENT]
        food = things[FOOD]

        the_plot.info['agent_position'] = agent.position
        the_plot.info['food_positions'] = np.stack(
            np.where(food.curtain), axis=-1)

        if food.curtain[agent.position]:
            the_plot.info['reached_food'] = True
            the_plot.add_reward(REWARD)
            food.curtain[agent.position] = False

        if np.sum(food.curtain) == 0.:
            the_plot.terminate_episode()


# Configures the trap tube environment.
TrapTubeConfig = collections.namedtuple(
    'TrapTubeConfig',
    ['art', 'tool_position', 'tool_size', 'tool_direction', 'food_position',
     'tool_category'])


class BaseTrapTubeEnv(gym_pycolab.PyColabEnv):
    """Trap Tube environment."""

    def __init__(self,
                 max_iterations=50,
                 delay=50,
                 resize_scale=32,
                 default_reward=0.):
        super(BaseTrapTubeEnv, self).__init__(
            max_iterations=max_iterations,
            default_reward=default_reward,
            action_space=spaces.MultiDiscrete(
                [len(Actions._fields), len(Directions._fields)]),
            resize_scale=resize_scale,
            delay=delay)

    @abc.abstractmethod
    def _make_trap_tube_config(self):
        """Create the game art.

        Returns:
            TrapTubeConfig.
        """
        pass

    def make_game(self):
        """Builds a trap tube game.

        Returns:
            pycolab.Engine
        """
        config = self._make_trap_tube_config()

        impassible_to_agent = SYMBOLIC_OBJECTS
        sprites = {
            AGENT: ascii_art.Partial(
                AgentSprite,
                impassible_to_agent)}
        drapes = {
            FOOD: ascii_art.Partial(
                FoodDrape,
                config.food_position),
            TRAP: TrapDrape,
            EXIT: ExitDrape,
            TUBE1: Tube1Drape,
            TUBE2: Tube2Drape,
            TOOL: ascii_art.Partial(
                ToolDrape,
                config.tool_position,
                tool_size=config.tool_size,
                tool_direction=config.tool_direction),
            TASK: TaskDrape}
        update_schedule = [
            [FOOD], [TOOL], [AGENT], [TUBE1, TUBE2], [TRAP], [EXIT], [TASK]]
        z_order = [
            TASK, TRAP, EXIT, TUBE1, TUBE2, FOOD, TOOL, AGENT]
        game = ascii_art.ascii_art_to_game(
            config.art,
            ' ',
            sprites,
            drapes,
            update_schedule=update_schedule,
            z_order=z_order,
            occlusion_in_layers=False)
        return game

    def make_colors(self):
        return {}


# Base config option.
base_config = TrapTubeConfig(
    art=[
        '            ',
        '            ',
        '            ',
        '            ',
        '    mmmm    ',
        '    u  n    ',
        '    u  n    ',
        '    wwww    ',
        '            ',
        ' a          ',
        '            ',
        '            ',
    ],
    tool_position=(4 + 2, 2),
    tool_size=4,
    tool_direction=0,
    food_position=(4 + 2, 4 + 1),
    tool_category=TOOL)

# Base colors option.
base_colors = {
    TOOL: TOOL_COLOR,
    AGENT: AGENT_COLOR,
    TUBE1: TUBE1_COLOR,
    TUBE2: TUBE2_COLOR,
    TRAP: TRAP_COLOR,
    EXIT: EXIT_COLOR,
    FOOD: FOOD_COLOR,
    GROUND: GROUND_COLOR,
}
