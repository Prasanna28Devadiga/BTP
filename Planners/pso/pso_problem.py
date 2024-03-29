# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

# pylint: disable=too-many-instance-attributes

import logging
from copy import deepcopy

from .particle import Particle
from ..util.problem_base import ProblemBase

LOGGER = logging.getLogger(__name__)


class PSOProblem(ProblemBase):
    def __init__(self, **kwargs):
        """
        Initialize a new particle swarm optimization problem.
        """
        super().__init__(**kwargs)
        self.__iteration_number = kwargs['iteration_number']
        self.__particles = [
            Particle(**kwargs, bit_generator=self._random)
            for _ in range(kwargs['particles'])
        ]

    def solve(self) -> Particle:
        # And also update global_best_particle
        best = None
        for iteration in range(self.__iteration_number):

            # Update global best
            global_best_particle = min(self.__particles)

            for particle in self.__particles:
                particle.step(global_best_particle.position)

            if not best or global_best_particle < best:
                best = deepcopy(global_best_particle)

            if self.iteration_callback:
                self.iteration_callback(iteration, best.position)

        LOGGER.info('Last best solution="%s"', best.value)
        return best
