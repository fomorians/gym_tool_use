"""Sprites & Things for tool use games."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
    
import collections

import numpy as np

from gym import spaces

import gym_pycolab

from pycolab import things as plab_things
from pycolab import ascii_art
from pycolab.prefab_parts import sprites as prefab_sprites


TOOL = 't'
TOOL_SIZE = 4
AGENT = 'a'
TUBE = 'x'
TRAP = 'u'
INVERTED_TRAP = 'n'
FOOD = 'f'
TASK = 'j'
GROUND = ' '
REWARD = 1.0

Actions = collections.namedtuple(
    'Actions', ['push', 'pull'])
Directions = collections.namedtuple(
    'Directions', ['up', 'down', 'left', 'right'])

ACTIONS = Actions(
    push=Directions(
        up=[1, 0, 0, 0, 0],
        down=[0, 1, 0, 0, 0],
        left=[0, 0, 1, 0, 0],
        right=[0, 0, 0, 1, 0],
    ),
    pull=Directions(
        up=[1, 0, 0, 0, 1],
        down=[0, 1, 0, 0, 1],
        left=[0, 0, 1, 0, 1],
        right=[0, 0, 0, 1, 1],
    )
)


class TubeDrape(plab_things.Drape):
    """Handles tube logic."""

    def update(self, actions, board, layers, backdrop, things, the_plot):
        pass


class TrapDrape(plab_things.Drape):
    """Handles trap logic."""

    def update(self, actions, board, layers, backdrop, things, the_plot):
        pass


class InvertedTrapDrape(plab_things.Drape):
    """Handles inverted trap logic."""

    def update(self, actions, board, layers, backdrop, things, the_plot):
        pass


# TODO(wenkesj): handle multiple foods.
class FoodDrape(plab_things.Drape):

    def __init__(self, curtain, character):
        super(FoodDrape, self).__init__(curtain, character)
        nz = np.nonzero(self.curtain)
        self._row, self._col = nz[0][0], nz[1][0]
        self.has_moved = False

    @property
    def position(self):
        return (self._row, self._col)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        """Handles food logic."""
        agent = things[AGENT]
        tool = things[TOOL]
        trap = things[TRAP]
        tube = things[TUBE]

        if not actions:
            return

        agent_row, agent_col = agent.position
        food_row, food_col = self.position

        if actions == ACTIONS.push.up:
            agent_is_south_of_tool = tool.is_south_of_tool(agent_row, agent_col)
            food_is_north_of_tool = tool.is_north_of_tool(food_row, food_col)
            if agent_is_south_of_tool and food_is_north_of_tool:
                if food_row <= 2:
                    self.has_moved = False
                else:
                    # trap_is_north_of_food = trap.curtain[food_row - 1, food_col]
                    trap_is_north_of_food = False
                    tube_is_north_of_food = tube.curtain[food_row - 1, food_col]
                    if trap_is_north_of_food or tube_is_north_of_food or (food_row == 0):
                        self.has_moved = False
                    else:
                        self._north(things)
            else:
                self.has_moved = False

        elif actions == ACTIONS.push.down:
            agent_is_north_of_tool = tool.is_north_of_tool(agent_row, agent_col)
            food_is_south_of_tool = tool.is_south_of_tool(food_row, food_col)
            if agent_is_north_of_tool and food_is_south_of_tool:
                if food_row >= (board.shape[0] - 2):
                    self.has_moved = False
                else:
                    # trap_is_south_of_food = trap.curtain[food_row + 1, food_col]
                    trap_is_south_of_food = False
                    tube_is_south_of_food = tube.curtain[food_row + 1, food_col]
                    if trap_is_south_of_food or tube_is_south_of_food:
                        self.has_moved = False
                    else:
                        self._south(things)
            else:
                self.has_moved = False

        elif actions == ACTIONS.push.right:
            agent_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(agent_row, agent_col)
            agent_is_west_of_tool = tool.is_west_of_tool(agent_row, agent_col)
            food_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(food_row, food_col)
            food_is_east_of_tool = tool.is_east_of_tool(food_row, food_col)
            if agent_is_west_of_tool and agent_is_in_row_range_of_tool and food_is_east_of_tool and food_is_in_row_range_of_tool:
                if food_col >= (board.shape[1] - 2):
                    self.has_moved = False
                else:
                    # trap_is_east_of_food = trap.curtain[food_row, food_col + 1]
                    trap_is_east_of_food = False
                    tube_is_east_of_food = tube.curtain[food_row, food_col + 1]
                    if trap_is_east_of_food or tube_is_east_of_food:
                        self.has_moved = False
                    else:
                        self._east(things)
            else:
                self.has_moved = False

        elif actions == ACTIONS.pull.left:
            agent_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(agent_row, agent_col)
            agent_is_west_of_tool = tool.is_west_of_tool(agent_row, agent_col)
            food_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(food_row, food_col)
            food_is_west_of_tool = tool.is_west_of_tool(food_row, food_col)
            if agent_is_west_of_tool and agent_is_in_row_range_of_tool and food_is_west_of_tool and food_is_in_row_range_of_tool:
                if food_col <= 3:
                    self.has_moved = False
                else:
                    # trap_is_west_of_food = trap.curtain[food_row, food_col - 1]
                    trap_is_west_of_food = False
                    tube_is_west_of_food = tube.curtain[food_row, food_col - 1]
                    if trap_is_west_of_food or tube_is_west_of_food:
                        self.has_moved = False
                    else:
                        self._west(things)
            else:
                self.has_moved = False

        elif actions == ACTIONS.push.left:
            agent_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(agent_row, agent_col)
            agent_is_east_of_tool = tool.is_east_of_tool(agent_row, agent_col)
            food_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(food_row, food_col)
            food_is_west_of_tool = tool.is_west_of_tool(food_row, food_col)
            if agent_is_east_of_tool and agent_is_in_row_range_of_tool and food_is_west_of_tool and food_is_in_row_range_of_tool:
                if food_col <= 3:
                    self.has_moved = False
                else:
                    # trap_is_west_of_food = trap.curtain[food_row, food_col - 1]
                    trap_is_west_of_food = False
                    tube_is_west_of_food = tube.curtain[food_row, food_col - 1]
                    if trap_is_west_of_food or tube_is_west_of_food:
                        self.has_moved = False
                    else:
                        self._west(things)
            else:
                self.has_moved = False

        elif actions == ACTIONS.pull.right:
            agent_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(agent_row, agent_col)
            agent_is_east_of_tool = tool.is_east_of_tool(agent_row, agent_col)
            food_is_in_row_range_of_tool = tool.is_in_row_range_of_tool(food_row, food_col)
            food_is_east_of_tool = tool.is_east_of_tool(food_row, food_col)
            if agent_is_east_of_tool and agent_is_in_row_range_of_tool and food_is_east_of_tool and food_is_in_row_range_of_tool:
                if food_col >= (board.shape[1] - 2):
                    self.has_moved = False
                else:
                    # trap_is_east_of_food = trap.curtain[food_row, food_col + 1]
                    trap_is_east_of_food = False
                    tube_is_east_of_food = tube.curtain[food_row, food_col + 1]
                    if trap_is_east_of_food or tube_is_east_of_food:
                        self.has_moved = False
                    else:
                        self._east(things)
            else:
                self.has_moved = False

        else:
            self._stay(things)

    def _north(self, things):
        self.curtain[self.position] = False
        self._row -= 1
        self.curtain[self.position] = True
        self.has_moved = True

    def _south(self, things):
        self.curtain[self.position] = False
        self._row += 1
        self.curtain[self.position] = True
        self.has_moved = True

    def _east(self, things):
        self.curtain[self.position] = False
        self._col += 1
        self.curtain[self.position] = True
        self.has_moved = True

    def _west(self, things):
        self.curtain[self.position] = False
        self._col -= 1
        self.curtain[self.position] = True
        self.has_moved = True

    def _stay(self, things):
        self.has_moved = False


class ToolDrape(plab_things.Drape):

    def __init__(self, curtain, character, position):
        super(ToolDrape, self).__init__(curtain, character)
        self._row, self._col = position
        assert (position[0] > 0) and ((position[0] + (TOOL_SIZE - 1)) < (curtain.shape[0] - 1))
        assert (position[1] > 0) and (position[1] < (curtain.shape[1] - 1))

        for position in [(row, self._col) for row in range(self._row, self._row + TOOL_SIZE)]:
            curtain[position] = True

    def _north(self, board, things):
        food = things[FOOD]
        row, col = food.position
        food_is_north_of_tool = self.is_north_of_tool(row, col)
        food_has_moved = food.has_moved
        if food_is_north_of_tool and not food_has_moved:
            return

        if self._row == 3:
            return

        curtain = np.roll(self.curtain, -1, axis=0)
        np.copyto(self.curtain, curtain)
        self._row -= 1

    def _south(self, board, things):
        food = things[FOOD]
        row, col = food.position
        food_is_south_of_tool = self.is_south_of_tool(row, col)
        food_has_moved = food.has_moved
        if food_is_south_of_tool and not food_has_moved:
            return

        if self._row == (board.shape[0] - 2):
            return

        curtain = np.roll(self.curtain, 1, axis=0)
        np.copyto(self.curtain, curtain)
        self._row += 1

    def _east(self, board, things):
        if self._col == (board.shape[1] - 3):
            return

        food = things[FOOD]
        row, col = food.position
        food_is_in_row_range_of_tool = self.is_in_row_range_of_tool(row, col)
        food_is_east_of_tool = self.is_east_of_tool(row, col)
        food_has_moved = food.has_moved
        if food_is_east_of_tool and food_is_in_row_range_of_tool and not food_has_moved:
            return

        curtain = np.roll(self.curtain, 1, axis=1)
        np.copyto(self.curtain, curtain)
        self._col += 1

    def _west(self, board, things):
        if self._col == 2:
            return

        food = things[FOOD]
        row, col = food.position
        food_is_in_row_range_of_tool = self.is_in_row_range_of_tool(row, col)
        food_is_west_of_tool = self.is_west_of_tool(row, col)
        food_has_moved = food.has_moved
        if food_is_west_of_tool and food_is_in_row_range_of_tool and not food_has_moved:
            return

        curtain = np.roll(self.curtain, -1, axis=1)
        np.copyto(self.curtain, curtain)
        self._col -= 1

    def _stay(self, board, things):
        pass

    def is_south_of_tool(self, row, col):
        contact_row = (self._row + TOOL_SIZE)
        is_south = (row == contact_row) and (col == self._col)
        return is_south

    def is_north_of_tool(self, row, col):
        return (row == (self._row - 1)) and (col == self._col)

    def is_west_of_tool(self, row, col):
        return col == (self._col - 1)

    def is_east_of_tool(self, row, col):
        return col == (self._col + 1)

    def is_in_row_range_of_tool(self, row, col):
        return self._row <= row <= (self._row + (TOOL_SIZE - 1))

    def update(self, actions, board, layers, backdrop, things, the_plot):
        """Handles tool logic."""
        agent = things[AGENT]

        if not actions:
            return

        row, col = agent.position
        if actions == ACTIONS.push.up:
            is_south_of_tool = self.is_south_of_tool(row, col)
            if is_south_of_tool:
                self._north(board, things)
        elif actions == ACTIONS.pull.down:
            is_south_of_tool = self.is_south_of_tool(row, col)
            if is_south_of_tool:
                self._south(board, things)
        elif actions == ACTIONS.pull.up:
            is_north_of_tool = self.is_north_of_tool(row, col)
            if is_north_of_tool:
                self._north(board, things)
        elif actions == ACTIONS.push.down:
            is_north_of_tool = self.is_north_of_tool(row, col)
            if is_north_of_tool:
                self._south(board, things)     
        elif actions == ACTIONS.push.right:
            is_in_row_range_of_tool = self.is_in_row_range_of_tool(row, col)
            is_west_of_tool = self.is_west_of_tool(row, col)
            if is_in_row_range_of_tool and is_west_of_tool:
                self._east(board, things)
        elif actions == ACTIONS.pull.left:
            is_in_row_range_of_tool = self.is_in_row_range_of_tool(row, col)
            is_west_of_tool = self.is_west_of_tool(row, col)
            if is_in_row_range_of_tool and is_west_of_tool:
                self._west(board, things)
        elif actions == ACTIONS.push.left:
            is_in_row_range_of_tool = self.is_in_row_range_of_tool(row, col)
            is_east_of_tool = self.is_east_of_tool(row, col)
            if is_in_row_range_of_tool and is_east_of_tool:
                self._west(board, things)
        elif actions == ACTIONS.pull.right:
            is_in_row_range_of_tool = self.is_in_row_range_of_tool(row, col)
            is_east_of_tool = self.is_east_of_tool(row, col)
            if is_in_row_range_of_tool and is_east_of_tool:
                self._east(board, things)
        else:
            self._stay(board, things)


class AgentSprite(prefab_sprites.MazeWalker):
    """Base agent sprite."""

    def __init__(self, corner, position, character):
        super(AgentSprite, self).__init__(
            corner, 
            position, 
            character, 
            TOOL + TUBE + TRAP,
            confined_to_board=True)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        """Handles agent logic."""
        rows, cols = self.position

        if not actions:
            return

        if actions[0]:    # go upward?
            self._north(board, the_plot)
        elif actions[1]:  # go downward?
            self._south(board, the_plot)
        elif actions[2]:  # go leftward?
            self._west(board, the_plot)
        elif actions[3]:  # go rightward?
            self._east(board, the_plot)
        else:
            self._stay(board, the_plot)


class TaskDrape(plab_things.Drape):
    """Handles task logic."""

    def update(self, actions, board, layers, backdrop, things, the_plot):
        agent = things[AGENT]
        food = things[FOOD]
        trap = things[TRAP]

        num_trapped_food = np.logical_and(
            trap.curtain, food.curtain)

        if np.sum(num_trapped_food) == np.sum(food.curtain):
            the_plot.terminate_episode()

        if food.curtain[agent.position]:
            the_plot.add_reward(REWARD)
            food.curtain[agent.position] = False

        if np.sum(food.curtain) == 0.:
            the_plot.terminate_episode()


class TrapTubeEnv(gym_pycolab.PyColabEnv):
    """Trap Tube environment."""

    base_art = [
        '        ', 
        '        ', 
        ' xx  xx ', 
        '        ', 
        '        ',
        ' xx  xx ', 
        '        ', 
        '        ',
    ]

    def __init__(self, max_iterations=50, art=None, tool_location=[3, 2]):
        # action_space:
        #     push(up):     [1, 0, 0, 0, 0]
        #     push(down):   [0, 1, 0, 0, 0]
        #     push(left):   [0, 0, 1, 0, 0]
        #     push(right):  [0, 0, 0, 1, 0]
        #     pull(up):     [1, 0, 0, 0, 1]
        #     pull(down):   [0, 1, 0, 0, 1]
        #     pull(left):   [0, 0, 1, 0, 1]
        #     pull(right):  [0, 0, 0, 1, 1]
        self.art = art
        self.tool_location = tool_location

        super(TrapTubeEnv, self).__init__(
            max_iterations=max_iterations, 
            default_reward=0.,
            action_space=spaces.MultiBinary(5),
            resize_scale=32,
            delay=200)

    def _make_art(self):
        # TODO(wenkesj): make art.
        return TrapTubeEnv.base_art

    def make_game(self):
        """Builds a trap tube game.

        Returns:
            pycolab.Engine
        """
        art = self.art if self.art else self._make_art()

        sprites = {
            AGENT: AgentSprite}
        drapes = {
            FOOD: FoodDrape, 
            TRAP: TrapDrape,
            INVERTED_TRAP: InvertedTrapDrape,
            TUBE: TubeDrape,
            TOOL: ascii_art.Partial(
                ToolDrape, self.tool_location),
            TASK: TaskDrape}
        update_schedule = [
            [FOOD],
            [TOOL], 
            [AGENT], 
            [TUBE], 
            [TRAP], 
            [INVERTED_TRAP],
            [TASK]]
        z_order = [
            TASK,
            INVERTED_TRAP, 
            TRAP, 
            TUBE,
            FOOD, 
            TOOL,
            AGENT]
        game = ascii_art.ascii_art_to_game(
            art, 
            ' ', 
            sprites, 
            drapes,
            update_schedule=update_schedule,
            z_order=z_order,
            occlusion_in_layers=False)
        return game

    def make_colors(self):
        return {
            TOOL:          (152, 208, 57),
            AGENT:         (113, 57, 208),
            TUBE:          (57, 152, 208),
            TRAP:          (208, 113, 57),
            INVERTED_TRAP: (208, 189, 57),
            FOOD:          (208, 57, 77),
            GROUND:        (72, 65, 17),
        }


if __name__ == "__main__":
    np.random.seed(42)

    env = TrapTubeEnv(
        art=[
            '          ', 
            '          ', 
            '          ', 
            '  xxnnxx  ', 
            '  n  f n  ', 
            '  n    n  ',
            '  xxnnxx  ', 
            '          ', 
            'a         ',
            '          ',
        ])
    lr_actions = [
        ACTIONS.push.right,
        ACTIONS.push.right,
        ACTIONS.push.up,
        ACTIONS.pull.down,
        ACTIONS.push.right,
        ACTIONS.push.up,
        ACTIONS.pull.right,
        ACTIONS.pull.right,
        ACTIONS.pull.right,
        ACTIONS.pull.right,
        ACTIONS.pull.right,
        ACTIONS.push.up,
        ACTIONS.push.up,
        ACTIONS.push.up,
        ACTIONS.push.up]
    ud_actions = [
        ACTIONS.push.right,
        ACTIONS.push.right,
        ACTIONS.push.up,
        ACTIONS.pull.down,
        ACTIONS.pull.down,
        ACTIONS.push.right,
        ACTIONS.push.up,
        ACTIONS.pull.right,
        ACTIONS.pull.right,
        ACTIONS.pull.right,
        ACTIONS.push.down,
        ACTIONS.push.left,
        ACTIONS.push.up,
        ACTIONS.push.up,
        ACTIONS.push.up,
        ACTIONS.push.right,
        ACTIONS.push.right,
        ACTIONS.push.right,
        ACTIONS.push.up,
        ACTIONS.push.up,
        ACTIONS.push.up,
        ACTIONS.push.up,
        ACTIONS.push.up,
        ACTIONS.push.left,
        ACTIONS.push.left,
        ACTIONS.push.left]

    for actions in [lr_actions, ud_actions]:
        state = env.reset()
        for action in actions:
            env.render()
            state, reward, done, _ = env.step(action)
            print(reward)
            if done:
                break
        env.render()
    env.close()