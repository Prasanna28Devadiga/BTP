# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

from math import gamma

import numpy as np


def levy_flight(start: np.ndarray, low_step: float, high_step: float, high_step_prob: float, gen: np.random.Generator) -> np.ndarray:
    """
    Perform a Lévy flight step.

    Arguments:
        start {numpy.ndarray} -- The cuckoo's start position
        low_step {float} -- step size of the low step
        high_step {float} -- step size of the low step
        high_step_prob -- probability of high step
        gen {Generator} -- the generator used to generate pseudo random numbers

    Returns:
        numpy.ndarray -- The new position
    """
    total_points = len(start) // 2
    point_change = list()

    for _ in range(total_points):
        if gen.random() <= high_step_prob:
            step_size = high_step
        else:
            step_size = low_step
        u = gen.random(2)
        u_hat = u / np.linalg.norm(u)
        v = u * step_size

        point_change.append(v[0])
        point_change.append(v[1])

    return start + np.array(point_change)
