from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from gym import spaces

from pycolab import ascii_art
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites

from gym_pycolab import pycolab_env


# Controls:
#   0
# 2 1 3

BRIDGE_BUILDING_TEMPLATE = [
    '########', 
    '#      #', 
    '#      #', 
    '#WWWWWW#',
    '#      #', 
    '#      #', 
    '#      #', 
    '########'
]

GOAL_REWARD = 1.
WATER_REWARD = -1.

BOXES = '1234567890'

COLORS = {
    'G': (255,   0,   0),
    'P': (  0, 255,   0),
    'W': (  0,   0, 255),
    '#': (  0, 255, 255),
}
BOX_COLORS = {box: (255, 255, 0) for box in BOXES}
COLORS.update(BOX_COLORS)

BOX_POSITIONS = [
    [(2, y) for y in range(1, len(BRIDGE_BUILDING_TEMPLATE) - 1)],
    [(x, y) 
     for y in range(1, len(BRIDGE_BUILDING_TEMPLATE) - 1)
     for x in range(4, 6)]
]


def paint(art, positions, characters):
    """Paint characters into their positions in the art."""
    for position, character in zip(positions, characters):
        art[position[0]][position[1]] = character
    return art


# TODO(wenkesj): ensure boxes are not generated ontop of 
# each other when there are is even number.
def generate_art(num_boxes):
    """Generate art given the number of boxes."""
    if num_boxes == -1:
        num_boxes = np.random.choice(3) + 1

    assert num_boxes > 0 and num_boxes < 4, '0 > `num_boxes` or `num_boxes` > 3 ' 
    x_positions = [1, len(BRIDGE_BUILDING_TEMPLATE) - 2]
    player_side, goal_side = np.random.choice(2, size=2, replace=False)

    # Generate player position
    player_position = (
        x_positions[player_side], 
        np.random.randint(1, len(BRIDGE_BUILDING_TEMPLATE) - 2))
    
    # Generate goal position
    goal_position = (
        x_positions[goal_side], 
        np.random.randint(1, len(BRIDGE_BUILDING_TEMPLATE) - 2))

    # Generate box positions
    box_side_positions = list(BOX_POSITIONS[player_side])
    box_position_indices = np.random.choice(
        len(box_side_positions), size=num_boxes, replace=False)
    box_positions = [box_side_positions[box_position_index] 
                     for box_position_index in box_position_indices]

    # Assign random numbers to the boxes.
    box_ids = np.random.choice(len(BOXES), size=num_boxes, replace=False)

    # Paint positions
    art = list(BRIDGE_BUILDING_TEMPLATE)
    art = [list(row) for row in art]
    art = paint(art, [player_position], ['P'])
    art = paint(art, [goal_position], ['G'])
    art = paint(art, box_positions, [str(box_id) for box_id in box_ids])

    # Random rotation
    art = np.rot90(art, np.random.choice(4))
    art = [''.join(row) for row in art.tolist()]
    return art


def make_game(level):
    """Builds and returns a Bridge Building game for the selected level."""
    bridge_building_art = generate_art(level)
    sprites = {'P': PlayerSprite}
    box_sprites = {box: BoxSprite for box in BOXES}
    sprites.update(box_sprites)
    drapes = {'G': GoalDrape, 'W': WaterDrape}
    return ascii_art.ascii_art_to_game(
        bridge_building_art, 
        ' ', 
        sprites, 
        drapes,
        update_schedule=[list(BOXES)] + [['P'], ['W'], ['G']],
        z_order=['G', 'W'] + list(BOXES) + ['P'],
        occlusion_in_layers=False)  # This allows water to exist underneath.


class WaterDrape(plab_things.Drape):

    def update(self, actions, board, layers, backdrop, things, the_plot):
        """Prevent player from moving through water."""
        player = things['P']
        row, col = player.position

        # End the episode if the agent is on the water tile.
        if self.curtain[player.position]:
            the_plot.add_reward(WATER_REWARD)
            the_plot.terminate_episode()


class GoalDrape(plab_things.Drape):

    def __init__(self, curtain, character):
        super(GoalDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        """Compute rewards from player interaction with goal tiles."""
        player = things['P']

        # Reward the agent if the goal has been reached and remove it.
        if self.curtain[player.position]:
            the_plot.add_reward(GOAL_REWARD)
            self.curtain[player.position] = False

        # End the episode if all goals have been reached.
        if np.sum(self.curtain) == 0.:
            the_plot.terminate_episode()


class BoxSprite(prefab_sprites.MazeWalker):

    def __init__(self, corner, position, character):
        impassable = set(BOXES + '#PGBX') - set(character)
        super(BoxSprite, self).__init__(corner, position, character, impassable)

        # Remove the boxes that aren't on the map.
        if position == (0, 0):
            self._teleport((-1, -1))

    def update(self, actions, board, layers, backdrop, things, the_plot):
        """Move boxes around and fill water spaces to create bridges."""
        del backdrop, things  # Unused.
        rows, cols = self.position

        # Don't move if already in water.
        if layers['W'][rows, cols]:
            self._stay(board, the_plot)
            return

        if actions == 0:    # go upward?
            if (rows + 1) < board.shape[0]:
                if layers['P'][rows + 1, cols]: 
                    self._north(board, the_plot)
            else:
                self._stay(board, the_plot)
        elif actions == 1:  # go downward?
            if (rows - 1) > 0:
                if layers['P'][rows - 1, cols]: 
                    self._south(board, the_plot)
            else:
                self._stay(board, the_plot)
        elif actions == 2:  # go leftward?
            if (cols + 1) < board.shape[1]:
                if layers['P'][rows, cols + 1]: 
                    self._west(board, the_plot)
            else:
                self._stay(board, the_plot)
        elif actions == 3:  # go rightward?
            if (cols - 1) > 0:
                if layers['P'][rows, cols - 1]: 
                    self._east(board, the_plot)
            else:
                self._stay(board, the_plot)


class PlayerSprite(prefab_sprites.MazeWalker):

    def __init__(self, corner, position, character):
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable=BOXES + '#X')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        """Moves the player and crosses bridges if possible."""
        del backdrop  # Unused.

        rows, cols = self.position
        if actions == 0:    # go upward?
            if (rows - 1) > 0:
                for box in BOXES:
                    box_is_north = things[box].position == (rows - 1, cols)
                    water_is_north = things['W'].curtain[rows - 1, cols]
                    can_cross_bridge = not layers['#'][rows - 2, cols]
                    if box_is_north and water_is_north:
                        if can_cross_bridge:  # cross the bridge?
                            self._teleport((self.virtual_position[0] - 2, 
                                            self.virtual_position[1] + 0))
                        else:
                            self._stay(board, the_plot)
                        return
            self._north(board, the_plot)
        elif actions == 1:  # go downward?
            if (rows + 1) < board.shape[0]:
                for box in BOXES:
                    box_is_south = things[box].position == (rows + 1, cols)
                    water_is_south = things['W'].curtain[rows + 1, cols]
                    can_cross_bridge = not layers['#'][rows + 2, cols] if (rows + 2) < board.shape[0] else False
                    if box_is_south and water_is_south:
                        if can_cross_bridge:  # cross the bridge?
                            self._teleport((self.virtual_position[0] + 2, 
                                            self.virtual_position[1] + 0))
                        else:
                            self._stay(board, the_plot)
                        return
            self._south(board, the_plot)
        elif actions == 2:  # go leftward?
            if (cols - 1) > 0:
                for box in BOXES:
                    box_is_west = things[box].position == (rows, cols - 1)
                    water_is_west = things['W'].curtain[rows, cols - 1]
                    can_cross_bridge = not layers['#'][rows, cols - 2] if (cols - 2) > 0 else False
                    if box_is_west and water_is_west:  
                        if can_cross_bridge:  # cross the bridge?
                            self._teleport((self.virtual_position[0] + 0, 
                                            self.virtual_position[1] - 2))
                        else:
                            self._stay(board, the_plot)
                        return
            self._west(board, the_plot)
        elif actions == 3:  # go rightward?
            if (cols + 1) < board.shape[1]:
                for box in BOXES:
                    box_is_east = things[box].position == (rows, cols + 1)
                    water_is_east = things['W'].curtain[rows, cols + 1]
                    can_cross_bridge = not layers['#'][rows, cols + 2] if (cols + 2) < board.shape[1] else False
                    if box_is_east and water_is_east:  
                        if can_cross_bridge:  # cross the bridge?
                            self._teleport((self.virtual_position[0] + 0, 
                                            self.virtual_position[1] + 2))
                        else:
                            self._stay(board, the_plot)
                        return
            self._east(board, the_plot)


class BridgeBuildingEnv(pycolab_env.PyColabEnv):
    """Bridge building game."""

    def __init__(self, 
                 level=1,
                 max_steps=10,
                 default_reward=0.):
        super(BridgeBuildingEnv, self).__init__(
            game_factory=lambda: make_game(level),
            max_iterations=max_steps, 
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            resize_scale=32,
            delay=200,
            colors=COLORS)
