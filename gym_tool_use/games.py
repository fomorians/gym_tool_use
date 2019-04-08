"""Tool use games."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pycolab import ascii_art

from gym_tool_use import sprites as tool_sprites
from gym_tool_use import things as tool_things
from gym_tool_use import utils


def make_tool_use_game(art, what_lies_beneath, prefill_positions):
    """Builds and returns a tool use.

    Args:
        art: the art of the game.
        what_lies_beneath: the art underneath the art.
        prefill_positions: dictionary mapping character (`Drape`) 
            to positions.

    Returns:
        a new game
    """

    # Include player.
    sprites = {'P': tool_sprites.PlayerSprite}

    # Include tools.
    box_sprites = {box: tool_sprites.BoxSprite for box in utils.BOXES}
    sprites.update(box_sprites)

    # Include the goal, water, and bridge.
    drapes = {
        'G': tool_things.GoalDrape, 
        'W': tool_things.WaterDrape,
        'B': tool_things.BridgeDrape}
    game = ascii_art.ascii_art_to_game(
        art, 
        what_lies_beneath, 
        sprites, 
        drapes,
        update_schedule=[list(utils.BOXES)] + [['P'], ['W'], ['B'], ['G']],
        z_order=['G', 'W'] + list(utils.BOXES) + ['B', 'P'],
        occlusion_in_layers=False)  # This allows layered representation.

    for character, positions in prefill_positions.items():
        layer = game._sprites_and_drapes[character]
        for position in positions:
            layer.curtain[position] = True

    return game