"""Trap tube environment."""

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

Grasps = collections.namedtuple(
    'Grasps', ['up', 'down', 'left', 'right'])
Movements = collections.namedtuple(
    'Movements', ['up', 'down', 'left', 'right'])

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3

ACTIONS = Grasps(
    up=Movements(
        up=[NORTH, NORTH],
        down=[NORTH, SOUTH],
        left=[NORTH, WEST],
        right=[NORTH, EAST]),
    down=Movements(
        up=[SOUTH, NORTH],
        down=[SOUTH, SOUTH],
        left=[SOUTH, WEST],
        right=[SOUTH, EAST]),
    left=Movements(
        up=[WEST, NORTH],
        down=[WEST, SOUTH],
        left=[WEST, WEST],
        right=[WEST, EAST]),
    right=Movements(
        up=[EAST, NORTH],
        down=[EAST, SOUTH],
        left=[EAST, WEST],
        right=[EAST, EAST]))


def _invert_direction(direction):
    if direction == NORTH:
        return SOUTH
    elif direction == SOUTH:
        return NORTH
    elif direction == WEST:
        return EAST
    elif direction == EAST:
        return WEST


def action_movement_equal(action, direction):
    return action[1] == direction


def _can_move(actions, row, col, board, things, impassables):
    if action_movement_equal(actions, NORTH):
        oob = row == 0
        row = row - 1
    elif action_movement_equal(actions, SOUTH):
        oob = row == (board.shape[0] - 1)
        row = row + 1
    elif action_movement_equal(actions, WEST):
        oob = col == 0
        col = col - 1
    elif action_movement_equal(actions, EAST):
        oob = col == (board.shape[1] - 1)
        col = col + 1
    else:
        return True

    if oob:
        return False

    for impassable in impassables:
        if things[impassable].curtain[row, col]:
            return False
    return True


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

    def can_move(self, actions, board, things, the_plot):
        agent = things[AGENT]
        tool = things[TOOL]

        agent_row, agent_col = agent.position
        food_row, food_col = self.position

        agent_can_grasp_tool = tool.check_adjacent(
            actions[0], agent_row, agent_col)
        agent_can_move = _can_move(
            actions, agent_row, agent_col, board, things,
            [TUBE1, TUBE2, TRAP, EXIT])
        agent_can_move_with_tool = agent_can_grasp_tool and agent_can_move

        food_in_movement = tool.check_adjacent(
            _invert_direction(actions[1]), food_row, food_col)
        food_can_move = _can_move(
            actions, food_row, food_col, board, things,
            [TOOL, TUBE1, TUBE2, TRAP])
        food_can_move_with_tool = food_can_move and food_in_movement
        return agent_can_move_with_tool and food_can_move_with_tool

    def update(self, actions, board, layers, backdrop, things, the_plot):
        """Handles food logic."""
        if actions is None:
            self.has_moved = False
            return

        if action_movement_equal(actions, NORTH):
            self._north(actions, board, things, the_plot)
        elif action_movement_equal(actions, SOUTH):
            self._south(actions, board, things, the_plot)
        elif action_movement_equal(actions, WEST):
            self._west(actions, board, things, the_plot)
        elif action_movement_equal(actions, EAST):
            self._east(actions, board, things, the_plot)
        else:
            self._stay(actions, board, things, the_plot)

    def _north(self, actions, board, things, the_plot):
        if self.can_move(actions, board, things, the_plot):
            the_plot.info['move_food_north'] = True
            self.curtain[self.position] = False
            self._row -= 1
            self.curtain[self.position] = True
            self.has_moved = True
        else:
            self.has_moved = False

    def _south(self, actions, board, things, the_plot):
        if self.can_move(actions, board, things, the_plot):
            the_plot.info['move_food_south'] = True
            self.curtain[self.position] = False
            self._row += 1
            self.curtain[self.position] = True
            self.has_moved = True
        else:
            self.has_moved = False

    def _east(self, actions, board, things, the_plot):
        if self.can_move(actions, board, things, the_plot):
            the_plot.info['move_food_east'] = True
            self.curtain[self.position] = False
            self._col += 1
            self.curtain[self.position] = True
            self.has_moved = True
        else:
            self.has_moved = False

    def _west(self, actions, board, things, the_plot):
        if self.can_move(actions, board, things, the_plot):
            the_plot.info['move_food_west'] = True
            self.curtain[self.position] = False
            self._col -= 1
            self.curtain[self.position] = True
            self.has_moved = True
        else:
            self.has_moved = False

    def _stay(self, actions, board, things, the_plot):
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

    def check_adjacent(self, direction, row, col):
        if direction == NORTH:
            is_in_col_range_of_tool = self.is_in_col_range_of_tool(
                row, col)
            is_south_of_tool = self.is_south_of_tool(
                row, col)
            return is_in_col_range_of_tool and is_south_of_tool
        elif direction == SOUTH:
            is_in_col_range_of_tool = self.is_in_col_range_of_tool(
                row, col)
            is_north_of_tool = self.is_north_of_tool(
                row, col)
            return is_in_col_range_of_tool and is_north_of_tool
        elif direction == WEST:
            is_in_row_range_of_tool = self.is_in_row_range_of_tool(
                row, col)
            is_east_of_tool = self.is_east_of_tool(
                row, col)
            return is_in_row_range_of_tool and is_east_of_tool
        elif direction == EAST:
            is_in_row_range_of_tool = self.is_in_row_range_of_tool(
                row, col)
            is_west_of_tool = self.is_west_of_tool(
                row, col)
            return is_in_row_range_of_tool and is_west_of_tool
        return False

    def _check_move(self, actions, board, things, the_plot):
        agent = things[AGENT]
        tool = things[TOOL]
        food = things[FOOD]

        agent_row, agent_col = agent.position
        food_row, food_col = food.position

        agent_did_grasp_tool = self.check_adjacent(
            actions[0], agent_row, agent_col)
        agent_can_move = _can_move(
            actions, agent_row, agent_col, board, things,
            [TUBE1, TUBE2, TRAP, EXIT])
        food_in_movement = self.check_adjacent(
            _invert_direction(actions[1]), food_row, food_col)
        food_has_moved = food.has_moved

        if not agent_did_grasp_tool:
            self.has_moved = False
            return False

        if not agent_can_move:
            self.has_moved = False
            agent.dont_move = True
            return False

        if food_in_movement and (not food_has_moved):
            self.has_moved = False
            agent.dont_move = True
            return False

        return True

    def _north(self, actions, board, things, the_plot):
        if self._check_move(actions, board, things, the_plot):
            if self._row == 0:
                self.has_moved = False
                things[AGENT].dont_move = True
                return

            the_plot.info['move_tool_north'] = True
            curtain = np.roll(self.curtain, -1, axis=0)
            np.copyto(self.curtain, curtain)
            self._row -= 1
            self.has_moved = True

    def _south(self, actions, board, things, the_plot):
        if self._check_move(actions, board, things, the_plot):
            if self._row == (board.shape[0] - 1 - (self._tool_size - 1) * int(
                    self._tool_direction == 0)):
                self.has_moved = False
                things[AGENT].dont_move = True
                return

            the_plot.info['move_tool_south'] = True
            curtain = np.roll(self.curtain, 1, axis=0)
            np.copyto(self.curtain, curtain)
            self._row += 1
            self.has_moved = True

    def _east(self, actions, board, things, the_plot):
        if self._check_move(actions, board, things, the_plot):
            if self._col == (board.shape[1] - 1 - (self._tool_size - 1) * int(
                    self._tool_direction == 1)):
                self.has_moved = False
                things[AGENT].dont_move = True
                return

            the_plot.info['move_tool_east'] = True
            curtain = np.roll(self.curtain, 1, axis=1)
            np.copyto(self.curtain, curtain)
            self._col += 1
            self.has_moved = True

    def _west(self, actions, board, things, the_plot):
        if self._check_move(actions, board, things, the_plot):
            if self._col == 0:
                self.has_moved = False
                things[AGENT].dont_move = True
                return

            the_plot.info['move_tool_west'] = True
            curtain = np.roll(self.curtain, -1, axis=1)
            np.copyto(self.curtain, curtain)
            self._col -= 1
            self.has_moved = True

    def _stay(self, actions, board, things, the_plot):
        self.has_moved = False

    def update(self, actions, board, layers, backdrop, things, the_plot):
        if actions is None:
            self.has_moved = False
            return

        if action_movement_equal(actions, NORTH):
            self._north(actions, board, things, the_plot)
        elif action_movement_equal(actions, SOUTH):
            self._south(actions, board, things, the_plot)
        elif action_movement_equal(actions, WEST):
            self._west(actions, board, things, the_plot)
        elif action_movement_equal(actions, EAST):
            self._east(actions, board, things, the_plot)
        else:
            self._stay(actions, board, things, the_plot)


class AgentSprite(prefab_sprites.MazeWalker):
    """Handles agent logic."""

    def __init__(self, corner, position, character, impassible):
        self.dont_move = False
        super(AgentSprite, self).__init__(
            corner,
            position,
            character,
            impassible,
            confined_to_board=True)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        tool = things[TOOL]
        rows, cols = self.position

        if actions is None:
            return

        if self.dont_move:
            self.dont_move = False
            self._stay(board, the_plot)
            return

        if action_movement_equal(actions, NORTH):
            self._north(board, the_plot)
        elif action_movement_equal(actions, SOUTH):
            self._south(board, the_plot)
        elif action_movement_equal(actions, WEST):
            self._west(board, the_plot)
        elif action_movement_equal(actions, EAST):
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
                 delay=250,
                 resize_scale=32,
                 default_reward=0.):
        super(BaseTrapTubeEnv, self).__init__(
            max_iterations=max_iterations,
            default_reward=default_reward,
            action_space=spaces.MultiDiscrete(
                [len(Grasps._fields), len(Movements._fields)]),
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

        sprites = {
            AGENT: ascii_art.Partial(
                AgentSprite,
                SYMBOLIC_OBJECTS)}
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
            [FOOD], [TOOL], [EXIT], [AGENT], [TUBE1, TUBE2], [TRAP], [TASK]]
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
    tool_position=(3 + 1, 1),
    tool_size=4,
    tool_direction=0,
    food_position=(4 + 1, 4 + 1),
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
