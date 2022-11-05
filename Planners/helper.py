import numpy as np


def cost_func(path, curr_map, weight_1, weight_2):
    """
    Returns the cost calculated as a weighted sum of the obstacle violations and the length of the path

    Args:
        path (array): array of pixels containing the path 
        curr_map (array): array of pixels containing obstacles
        weight_1 (float): weight given to obstacle avoidance
        weight_2 (float): weight given to shortest length
    """
    total_weight = weight_1 + weight_2
    weight_1 /= total_weight
    weight_2 /= total_weight

    length_of_path = 0
    violation = 0
    prev_x, prev_y = path[0]
    root_2 = 2 ** 0.5
    for curr_x, curr_y in path[1:]:
        if curr_x == prev_x or curr_y == prev_y:
            length_of_path += 1
        else:
            length_of_path += root_2
        curr_cost = curr_map[curr_y, curr_x]
        if curr_cost == 0:
            violation += 1
    
    cost = weight_1 * violation + weight_2 * length_of_path

    return cost                
    

def calc_dist(p1, p2):
    """
    Returns distance between two points

    Args:
        p1 (tuple): Point 1
        p2 (tuple): Point 2
    """
    dist = np.linalg.norm(p1-p2)
    return dist


def draw_line(start, end):
    itr = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
    Px = np.linspace(start[0], end[0], itr + 1)
    Py = np.linspace(start[1], end[1], itr + 1)
    final_points = np.concatenate((Px.reshape(-1, 1), Py.reshape(-1, 1)), axis=1)
    return final_points


def linear_interpolation(start, end, inter_points):
    """
    Returns the straight path between start and goal position
    """
    prev_point = start
    inter_points = np.append(inter_points, np.array([end]), axis=0)
    path = None
    for point in inter_points:
        curr_path = draw_line(prev_point, point)
        if path is None:
            path = curr_path
        else:
            path = np.append(path, curr_path[1:], axis=0)
        prev_point = point
    return path.round().astype(np.int32)
