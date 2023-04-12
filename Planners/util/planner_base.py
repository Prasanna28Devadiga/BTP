from ..helper import linear_interpolation, cost_func, cost_func2
import numpy as np


def get_cost_function(curr_map, start, end, cost_func_wt):
    def curr_cost_func(particles):
        points = particles.reshape(-1, 2)
        points = (points * (np.array(curr_map.shape[::-1]) - 1)).astype(np.int32)
        path = linear_interpolation(start, end, points)
        cost = cost_func(path, curr_map, cost_func_wt[0], cost_func_wt[1])
        return cost
    return curr_cost_func


def get_cost_function2(curr_map, start, end, cost_func_wt, com_points, com_weights):
    def curr_cost_func(particles):
        points = particles.reshape(-1, 2)
        points = (points * (np.array(curr_map.shape[::-1]) - 1)).astype(np.int32)
        path = linear_interpolation(start, end, points)
        cost = cost_func2(path, curr_map, com_points, com_weights, cost_func_wt[0], cost_func_wt[1], cost_func_wt[2])
        return cost
    return curr_cost_func


class PlannerBase:
    def __init__(self, **kwargs):
        self.map = kwargs['map']
        self.optimizer_params = kwargs['optimizer_params']
        self.optimizer = kwargs['optimizer']
        self.cost_func_wt = kwargs['cost_func_wt']
        self.function2 = False
        if 'com_points' in kwargs:
            self.com_points = kwargs['com_points']
            self.com_weights = kwargs['com_weights']
            self.function2 = True
        if 'initial_particle' in self.optimizer_params:
            self.optimizer_params['one_set'] = [True]

    def get_path(self, start, end):
        if self.function2:
            cost_function = get_cost_function2(
                self.map,
                start,
                end,
                self.cost_func_wt,
                self.com_points,
                self.com_weights
            )
        else:
            cost_function = get_cost_function(
                self.map,
                start,
                end,
                self.cost_func_wt
            )

        opt = self.optimizer(**self.optimizer_params, function=cost_function)
        best_sol = opt.solve()

        particles = best_sol.position
        points = particles.reshape(-1, 2)
        points = (points * (np.array(self.map.shape[::-1]) - 1)).astype(np.int32)
        path = linear_interpolation(start, end, points)

        return path
