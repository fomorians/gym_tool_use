"""Tool use games."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pycolab import ascii_art

from gym_tool_use import sprites as tool_sprites
from gym_tool_use import things as tool_things
from gym_tool_use import utils


def make_tool_use_game(art):
    """Builds and returns a tool use."""
    # Include player.
    sprites = {'P': tool_sprites.PlayerSprite}

    # Include tools.
    water_box_sprites = {box: tool_sprites.WaterBoxSprite for box in utils.WATER_BOXES}
    box_sprites = {box: tool_sprites.BoxSprite for box in utils.BOXES}
    sprites.update(water_box_sprites)
    sprites.update(box_sprites)

    # Include the goal and water.
    drapes = {'G': tool_things.GoalDrape, 'W': tool_things.WaterDrape}
    return ascii_art.ascii_art_to_game(
        art, 
        ' ', 
        sprites, 
        drapes,
        update_schedule=[list(utils.BOXES)] + [['P'], ['W'], ['G']],
        z_order=['G', 'W'] + list(utils.BOXES) + ['P'],
        occlusion_in_layers=False)  # This allows anything to exist underneath.
