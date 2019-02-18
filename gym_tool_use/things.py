"""Things for tool use games."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pycolab import things as plab_things


class WaterDrape(plab_things.Drape):
    """Handles water logic."""

    def update(self, actions, board, layers, backdrop, things, the_plot):
        """Prevent player from moving through water."""
        player = things['P']
        row, col = player.position

        # End the episode if the agent is on the water tile.
        if self.curtain[player.position]:
            the_plot.add_reward(-1.)
            the_plot.terminate_episode()


class GoalDrape(plab_things.Drape):
    """Handles goal logic."""

    def __init__(self, curtain, character):
        super(GoalDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        """Compute rewards from player interaction with goal tiles."""
        player = things['P']

        # Reward the agent if the goal has been reached and remove it.
        if self.curtain[player.position]:
            the_plot.add_reward(1.)
            self.curtain[player.position] = False

        # End the episode if all goals have been reached.
        if np.sum(self.curtain) == 0.:
            the_plot.terminate_episode()
