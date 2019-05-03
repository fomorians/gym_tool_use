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
FAKE_TOOL = 'q'
AGENT = 'a'
TUBE = 'm'
FAKE_TUBE = 'w'
TRAP = 'u'
FAKE_TRAP = 'n'
FOOD = 'f'
TASK = 'j'
GROUND = ' '
TOOL_COLOR = (152, 208, 57)
FAKE_TOOL_COLOR = (152, 255, 57),
AGENT_COLOR = (113, 57, 208)
TUBE_COLOR = (57, 152, 208)
FAKE_TUBE_COLOR = (57, 202, 208)
TRAP_COLOR = (208, 113, 57)
FAKE_TRAP_COLOR = (208, 189, 57)
FOOD_COLOR = (208, 57, 77)
GROUND_COLOR = (72, 65, 17)
REWARD = 1.0

Actions = collections.namedtuple(
    'Actions', ['push', 'pull', 'move'])
Directions = collections.namedtuple(
    'Directions', ['up', 'down', 'left', 'right'])

ACTIONS = Actions(
    push=Directions(
        up=0,
        down=1,
        left=2,
        right=3),
    pull=Directions(
        up=4,
        down=5,
        left=6,
        right=7),
    move=Directions(
        up=8,
        down=9,
        left=10,
        right=11))


class TubeDrape(plab_things.Drape):
    """Handles tube logic."""

    def update(self, actions, board, layers, backdrop, things, the_plot):
        pass


class FakeTubeDrape(plab_things.Drape):
    """Handles fake tube logic."""

    def update(self, actions, board, layers, backdrop, things, the_plot):
        pass


class TrapDrape(plab_things.Drape):
    """Handles trap logic."""

    def update(self, actions, board, layers, backdrop, things, the_plot):
        pass


class FakeTrapDrape(plab_things.Drape):
    """Handles fake trap logic."""

    def update(self, actions, board, layers, backdrop, things, the_plot):
        pass


# TODO(wenkesj): handle multiple foods.
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
        tube = things[TUBE]
        fake_tube = things[FAKE_TUBE]

        if not actions:
            self.has_moved = False
            return

        agent_row, agent_col = agent.position
        food_row, food_col = self.position

        if actions == ACTIONS.push.up:
            agent_is_in_col_range_of_tool = tool.is_in_col_range_of_tool(
                agent_row, agent_col)
            agent_is_south_of_tool = tool.is_south_of_tool(
                agent_row, agent_col)
            food_is_in_col_range_of_tool = tool.is_in_col_range_of_tool(
                food_row, food_col)
            food_is_north_of_tool = tool.is_north_of_tool(food_row, food_col)
            if agent_is_south_of_tool and agent_is_in_col_range_of_tool and \
                    food_is_north_of_tool and food_is_in_col_range_of_tool:
                # TODO(wenkesj): Trap falling behavior.
                trap_is_north_of_food = trap.curtain[
                    food_row - 1, food_col]
                # trap_is_north_of_food = False

                tube_is_north_of_food = tube.curtain[
                    food_row - 1, food_col]
                fake_tube_is_north_of_food = fake_tube.curtain[
                    food_row - 1, food_col]
                if trap_is_north_of_food or tube_is_north_of_food or \
                        (food_row == 0) or fake_tube_is_north_of_food:
                    self.has_moved = False
                else:
                    self._north(board, things)
            else:
                self.has_moved = False

        if actions == ACTIONS.pull.up:
            agent_is_in_col_range_of_tool = tool.is_in_col_range_of_tool(
                agent_row, agent_col)
            agent_is_north_of_tool = tool.is_north_of_tool(
                agent_row, agent_col)
            food_is_in_col_range_of_tool = tool.is_in_col_range_of_tool(
                food_row, food_col)
            food_is_north_of_tool = tool.is_north_of_tool(food_row, food_col)
            if agent_is_north_of_tool and agent_is_in_col_range_of_tool and \
                    food_is_north_of_tool and food_is_in_col_range_of_tool:
                # TODO(wenkesj): Trap falling behavior.
                trap_is_north_of_food = trap.curtain[
                    food_row - 1, food_col]
                # trap_is_north_of_food = False

                tube_is_north_of_food = tube.curtain[
                    food_row - 1, food_col]
                fake_tube_is_north_of_food = fake_tube.curtain[
                    food_row - 1, food_col]
                if trap_is_north_of_food or tube_is_north_of_food or \
                        (food_row == 0) or fake_tube_is_north_of_food:
                    self.has_moved = False
                else:
                    self._north(board, things)
            else:
                self.has_moved = False

        elif actions == ACTIONS.push.down:
            agent_is_in_col_range_of_tool = tool.is_in_col_range_of_tool(
                agent_row, agent_col)
            agent_is_north_of_tool = tool.is_north_of_tool(
                agent_row, agent_col)
            food_is_in_col_range_of_tool = tool.is_in_col_range_of_tool(
                food_row, food_col)
            food_is_south_of_tool = tool.is_south_of_tool(food_row, food_col)
            if agent_is_north_of_tool and agent_is_in_col_range_of_tool and \
                    food_is_south_of_tool and food_is_in_col_range_of_tool:
                # TODO(wenkesj): Trap falling behavior.
                trap_is_south_of_food = trap.curtain[
                    food_row + 1, food_col]
                # trap_is_south_of_food = False

                tube_is_south_of_food = tube.curtain[
                    food_row + 1, food_col]
                fake_tube_is_south_of_food = fake_tube.curtain[
                    food_row + 1, food_col]
                if trap_is_south_of_food or tube_is_south_of_food or \
                        fake_tube_is_south_of_food:
                    self.has_moved = False
                else:
                    self._south(board, things)
            else:
                self.has_moved = False

        elif actions == ACTIONS.pull.down:
            agent_is_in_col_range_of_tool = tool.is_in_col_range_of_tool(
                agent_row, agent_col)
            agent_is_south_of_tool = tool.is_south_of_tool(
                agent_row, agent_col)
            food_is_in_col_range_of_tool = tool.is_in_col_range_of_tool(
                food_row, food_col)
            food_is_south_of_tool = tool.is_south_of_tool(food_row, food_col)
            if agent_is_south_of_tool and agent_is_in_col_range_of_tool and \
                    food_is_south_of_tool and food_is_in_col_range_of_tool:
                # TODO(wenkesj): Trap falling behavior.
                trap_is_south_of_food = trap.curtain[
                    food_row + 1, food_col]
                # trap_is_south_of_food = False

                tube_is_south_of_food = tube.curtain[
                    food_row + 1, food_col]
                fake_tube_is_south_of_food = fake_tube.curtain[
                    food_row + 1, food_col]
                if trap_is_south_of_food or tube_is_south_of_food or \
                        fake_tube_is_south_of_food:
                    self.has_moved = False
                else:
                    self._south(board, things)
            else:
                self.has_moved = False

        elif actions == ACTIONS.push.right:
            agent_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(
                agent_row, agent_col)
            agent_is_west_of_tool = tool.is_west_of_tool(agent_row, agent_col)
            food_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(
                food_row, food_col)
            food_is_east_of_tool = tool.is_east_of_tool(food_row, food_col)
            if agent_is_west_of_tool and agent_is_in_row_range_of_tool and \
                    food_is_east_of_tool and food_is_in_row_range_of_tool:
                # TODO(wenkesj): Trap falling behavior.
                trap_is_east_of_food = trap.curtain[food_row, food_col + 1]
                # trap_is_east_of_food = False

                tube_is_east_of_food = tube.curtain[food_row, food_col + 1]
                fake_tube_is_east_of_food = fake_tube.curtain[
                    food_row, food_col + 1]
                if trap_is_east_of_food or tube_is_east_of_food or \
                        fake_tube_is_east_of_food:
                    self.has_moved = False
                else:
                    self._east(board, things)
            else:
                self.has_moved = False

        elif actions == ACTIONS.pull.left:
            agent_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(
                agent_row, agent_col)
            agent_is_west_of_tool = tool.is_west_of_tool(agent_row, agent_col)
            food_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(
                food_row, food_col)
            food_is_west_of_tool = tool.is_west_of_tool(food_row, food_col)
            if agent_is_west_of_tool and agent_is_in_row_range_of_tool and \
                    food_is_west_of_tool and food_is_in_row_range_of_tool:
                # TODO(wenkesj): Trap falling behavior.
                trap_is_west_of_food = trap.curtain[food_row, food_col - 1]
                # trap_is_west_of_food = False

                tube_is_west_of_food = tube.curtain[food_row, food_col - 1]
                fake_tube_is_west_of_food = fake_tube.curtain[
                    food_row, food_col - 1]
                if trap_is_west_of_food or tube_is_west_of_food or \
                        fake_tube_is_west_of_food:
                    self.has_moved = False
                else:
                    self._west(board, things)
            else:
                self.has_moved = False

        elif actions == ACTIONS.push.left:
            agent_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(
                agent_row, agent_col)
            agent_is_east_of_tool = tool.is_east_of_tool(agent_row, agent_col)
            food_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(
                food_row, food_col)
            food_is_west_of_tool = tool.is_west_of_tool(food_row, food_col)
            if agent_is_east_of_tool and agent_is_in_row_range_of_tool and \
                    food_is_west_of_tool and food_is_in_row_range_of_tool:
                # TODO(wenkesj): Trap falling behavior.
                trap_is_west_of_food = trap.curtain[food_row, food_col - 1]
                # trap_is_west_of_food = False
                tube_is_west_of_food = tube.curtain[food_row, food_col - 1]
                fake_tube_is_west_of_food = fake_tube.curtain[
                    food_row, food_col - 1]
                if trap_is_west_of_food or tube_is_west_of_food or \
                        fake_tube_is_west_of_food:
                    self.has_moved = False
                else:
                    self._west(board, things)
            else:
                self.has_moved = False

        elif actions == ACTIONS.pull.right:
            agent_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(
                agent_row, agent_col)
            agent_is_east_of_tool = tool.is_east_of_tool(agent_row, agent_col)
            food_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(
                food_row, food_col)
            food_is_east_of_tool = tool.is_east_of_tool(food_row, food_col)
            if agent_is_east_of_tool and agent_is_in_row_range_of_tool and \
                    food_is_east_of_tool and food_is_in_row_range_of_tool:
                # TODO(wenkesj): Trap falling behavior.
                trap_is_east_of_food = trap.curtain[food_row, food_col + 1]
                # trap_is_east_of_food = False

                tube_is_east_of_food = tube.curtain[food_row, food_col + 1]
                fake_tube_is_east_of_food = fake_tube.curtain[
                    food_row, food_col + 1]
                if trap_is_east_of_food or tube_is_east_of_food or \
                        fake_tube_is_east_of_food:
                    self.has_moved = False
                else:
                    self._east(board, things)
            else:
                self.has_moved = False

        else:
            self._stay(board, things)

    def _north(self, board, things):
        if self._row > 0:
            self.curtain[self.position] = False
            self._row -= 1
            self.curtain[self.position] = True
            self.has_moved = True
        else:
            self.has_moved = False

    def _south(self, board, things):
        if self._row < board.shape[0]:
            self.curtain[self.position] = False
            self._row += 1
            self.curtain[self.position] = True
            self.has_moved = True
        else:
            self.has_moved = False

    def _east(self, board, things):
        if self._col < board.shape[1]:
            self.curtain[self.position] = False
            self._col += 1
            self.curtain[self.position] = True
            self.has_moved = True
        else:
            self.has_moved = False

    def _west(self, board, things):
        if self._col > 0:
            self.curtain[self.position] = False
            self._col -= 1
            self.curtain[self.position] = True
            self.has_moved = True
        else:
            self.has_moved = False

    def _stay(self, board, things):
        self.has_moved = False


class BaseToolDrape(plab_things.Drape):
    """Handles base tool logic."""

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

        super(BaseToolDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        pass


class FakeToolDrape(BaseToolDrape):
    """Handles fake tool logic."""
    pass


# TODO(wenkesj): handle multiple tools.
class ToolDrape(BaseToolDrape):
    """Handles tool logic."""

    def _north(self, board, things):
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

        curtain = np.roll(self.curtain, -1, axis=0)
        np.copyto(self.curtain, curtain)
        self._row -= 1
        self.has_moved = True

    def _south(self, board, things):
        if self._row == (board.shape[1] - 1):
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

        curtain = np.roll(self.curtain, 1, axis=0)
        np.copyto(self.curtain, curtain)
        self._row += 1
        self.has_moved = True

    def _east(self, board, things):
        if self._col == (board.shape[1] - 1):
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

        curtain = np.roll(self.curtain, 1, axis=1)
        np.copyto(self.curtain, curtain)
        self._col += 1
        self.has_moved = True

    def _west(self, board, things):
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

        curtain = np.roll(self.curtain, -1, axis=1)
        np.copyto(self.curtain, curtain)
        self._col -= 1
        self.has_moved = True

    def _stay(self, board, things):
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

        if not actions:
            self.has_moved = False
            return

        row, col = agent.position
        if actions == ACTIONS.push.up:
            is_in_col_range_of_tool = self.is_in_col_range_of_tool(row, col)
            is_south_of_tool = self.is_south_of_tool(row, col)
            if is_in_col_range_of_tool and is_south_of_tool:
                self._north(board, things)
            else:
                self.has_moved = False
        elif actions == ACTIONS.pull.down:
            is_in_col_range_of_tool = self.is_in_col_range_of_tool(row, col)
            is_south_of_tool = self.is_south_of_tool(row, col)
            if is_in_col_range_of_tool and is_south_of_tool:
                self._south(board, things)
            else:
                self.has_moved = False
        elif actions == ACTIONS.pull.up:
            is_in_col_range_of_tool = self.is_in_col_range_of_tool(row, col)
            is_north_of_tool = self.is_north_of_tool(row, col)
            if is_in_col_range_of_tool and is_north_of_tool:
                self._north(board, things)
            else:
                self.has_moved = False
        elif actions == ACTIONS.push.down:
            is_in_col_range_of_tool = self.is_in_col_range_of_tool(row, col)
            is_north_of_tool = self.is_north_of_tool(row, col)
            if is_in_col_range_of_tool and is_north_of_tool:
                self._south(board, things)
            else:
                self.has_moved = False
        elif actions == ACTIONS.push.right:
            is_in_row_range_of_tool = self.is_in_row_range_of_tool(row, col)
            is_west_of_tool = self.is_west_of_tool(row, col)
            if is_in_row_range_of_tool and is_west_of_tool:
                self._east(board, things)
            else:
                self.has_moved = False
        elif actions == ACTIONS.pull.left:
            is_in_row_range_of_tool = self.is_in_row_range_of_tool(row, col)
            is_west_of_tool = self.is_west_of_tool(row, col)
            if is_in_row_range_of_tool and is_west_of_tool:
                self._west(board, things)
            else:
                self.has_moved = False
        elif actions == ACTIONS.push.left:
            is_in_row_range_of_tool = self.is_in_row_range_of_tool(row, col)
            is_east_of_tool = self.is_east_of_tool(row, col)
            if is_in_row_range_of_tool and is_east_of_tool:
                self._west(board, things)
            else:
                self.has_moved = False
        elif actions == ACTIONS.pull.right:
            is_in_row_range_of_tool = self.is_in_row_range_of_tool(row, col)
            is_east_of_tool = self.is_east_of_tool(row, col)
            if is_in_row_range_of_tool and is_east_of_tool:
                self._east(board, things)
            else:
                self.has_moved = False
        else:
            self._stay(board, things)


class AgentSprite(prefab_sprites.MazeWalker):
    """Handles agent logic."""

    def __init__(self, corner, position, character, impassible):
        super(AgentSprite, self).__init__(
            corner,
            position,
            character,
            impassible,
            confined_to_board=True)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        tool = things[TOOL]
        rows, cols = self.position

        if not actions:
            return

        if actions == ACTIONS.move.up:
            self._north(board, the_plot)
        elif actions == ACTIONS.push.up:
            if tool.has_moved:
                self._north(board, the_plot)
        elif actions == ACTIONS.pull.up:
            if tool.has_moved:
                self._north(board, the_plot)
        elif actions == ACTIONS.move.down:
            self._south(board, the_plot)
        elif actions == ACTIONS.push.down:
            if tool.has_moved:
                self._south(board, the_plot)
        elif actions == ACTIONS.pull.down:
            if tool.has_moved:
                self._south(board, the_plot)
        elif actions == ACTIONS.move.left:
            self._west(board, the_plot)
        elif actions == ACTIONS.push.left:
            if tool.has_moved:
                self._west(board, the_plot)
        elif actions == ACTIONS.pull.left:
            if tool.has_moved:
                self._west(board, the_plot)
        elif actions == ACTIONS.move.right:
            self._east(board, the_plot)
        elif actions == ACTIONS.push.right:
            if tool.has_moved:
                self._east(board, the_plot)
        elif actions == ACTIONS.pull.right:
            if tool.has_moved:
                self._east(board, the_plot)
        else:
            self._stay(board, the_plot)


class TaskDrape(plab_things.Drape):
    """Handles task logic."""

    def update(self, actions, board, layers, backdrop, things, the_plot):
        agent = things[AGENT]
        food = things[FOOD]

        # # TODO(wenkesj): Trap falling behavior. End episode of the food has
        # # "fallen" into trap.
        # trap = things[TRAP]
        # num_trapped_food = np.logical_and(
        #     trap.curtain, food.curtain)
        # if np.sum(num_trapped_food) == np.sum(food.curtain):
        #     the_plot.terminate_episode()

        if food.curtain[agent.position]:
            the_plot.add_reward(REWARD)
            food.curtain[agent.position] = False

        if np.sum(food.curtain) == 0.:
            the_plot.terminate_episode()


# Configures the trap tube environment.
TrapTubeConfig = collections.namedtuple(
    'TrapTubeConfig',
    ['art', 'tool_position', 'tool_size', 'tool_direction',
     'fake_tool_position', 'fake_tool_size', 'fake_tool_direction',
     'food_position'])


class BaseTrapTubeEnv(gym_pycolab.PyColabEnv):
    """Trap Tube environment."""

    def __init__(self, max_iterations=50):
        super(BaseTrapTubeEnv, self).__init__(
            max_iterations=max_iterations,
            default_reward=0.,
            action_space=spaces.Discrete(
                len(Actions._fields) * len(Directions._fields)),
            resize_scale=32,
            delay=60)

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

        impassible_to_agent = TUBE + TOOL + FAKE_TOOL + TRAP + FAKE_TRAP
        sprites = {
            AGENT: ascii_art.Partial(
                AgentSprite,
                impassible_to_agent)}
        drapes = {
            FOOD: ascii_art.Partial(
                FoodDrape,
                config.food_position),
            TRAP: TrapDrape,
            FAKE_TRAP: FakeTrapDrape,
            TUBE: TubeDrape,
            FAKE_TUBE: FakeTubeDrape,
            TOOL: ascii_art.Partial(
                ToolDrape,
                config.tool_position,
                tool_size=config.tool_size,
                tool_direction=config.tool_direction),
            FAKE_TOOL: ascii_art.Partial(
                FakeToolDrape,
                config.fake_tool_position,
                tool_size=config.fake_tool_size,
                tool_direction=config.fake_tool_direction),
            TASK: TaskDrape}
        update_schedule = [
            [FOOD], [TOOL], [FAKE_TOOL], [AGENT], [TUBE], [FAKE_TUBE], [TRAP],
            [FAKE_TRAP], [TASK]]
        z_order = [
            TASK, TRAP, FAKE_TRAP, TUBE, FAKE_TUBE, FOOD, TOOL, FAKE_TOOL,
            AGENT]
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
        '          ',
        '          ',
        '          ',
        '  mmmmmm  ',
        '  u    n  ',
        '  u    n  ',
        '  mmmmmm  ',
        '          ',
        'a         ',
        '          ',
    ],
    tool_position=(3, 3),
    tool_size=4,
    tool_direction=0,
    fake_tool_position=(-1, -1),
    fake_tool_size=0,
    fake_tool_direction=0,
    food_position=(4, 4))

# Base colors option.
base_colors = {
    TOOL: TOOL_COLOR,
    FAKE_TOOL: FAKE_TOOL_COLOR,
    AGENT: AGENT_COLOR,
    TUBE: TUBE_COLOR,
    FAKE_TUBE: FAKE_TUBE_COLOR,
    TRAP: TRAP_COLOR,
    FAKE_TRAP: FAKE_TRAP_COLOR,
    FOOD: FOOD_COLOR,
    GROUND: GROUND_COLOR,
}
