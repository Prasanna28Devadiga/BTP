import numpy as np


def get_next_coordinates(x, y, max_x, max_y, visited=None):
    if visited is None:
        visited = np.zeros((max_y, max_x), dtype=bool)
    next_points = list()
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            new_x = x + dx
            new_y = y + dy

            if 0 <= new_x < max_x and 0 <= new_y < max_y and not visited[new_y, new_x]:
                next_points.append((new_x, new_y))
    return next_points


def get_com_points(curr_map):
    visited = np.zeros(curr_map.shape, dtype=bool)
    com_points = list()
    com_weights = list()
    for y in range(curr_map.shape[0]):
        for x in range(curr_map.shape[1]):
            if visited[y, x]:
                continue
            else:
                visited[y, x] = True

            if curr_map[y, x] != 0:
                continue
            obstacle_points = [(x, y)]
            obstacle_weight = 1
            frontier = get_next_coordinates(x, y, curr_map.shape[1], curr_map.shape[0], visited)
            while len(frontier) > 0:
                curr_point, frontier = frontier[0], frontier[1:]
                if visited[curr_point[1], curr_point[0]]:
                    continue
                else:
                    visited[curr_point[1], curr_point[0]] = True

                if curr_map[curr_point[1], curr_point[0]] == 0:
                    obstacle_points.append(curr_point)
                    obstacle_weight += 1
                    frontier += get_next_coordinates(*curr_point, curr_map.shape[1], curr_map.shape[0], visited)
            obstacle_points = np.array(obstacle_points)
            curr_com = np.mean(obstacle_points, axis=0)
            com_points.append(np.round(curr_com).astype(int))
            com_weights.append(obstacle_weight)
    return np.array(com_points), np.array(com_weights)


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


def cost_func2(path, curr_map, com_points, com_weights, weight_1, weight_2, weight_3):
    total_weight = weight_1 + weight_2 + weight_3
    weight_1 /= total_weight
    weight_2 /= total_weight
    weight_3 /= total_weight

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

    com_cost = 0
    for i in range(len(com_points)):
        com_x, com_y = com_points[i, 0], com_points[i, 1]
        com_mass = com_weights[i]
        min_distance = np.inf
        for curr_x, curr_y in path:
            curr_distance = np.sqrt((com_x - curr_x)**2 + (com_y - curr_y)**2)
            if curr_distance < min_distance:
                min_distance = curr_distance
        com_cost += (com_mass / min_distance)

    cost = weight_1 * violation + weight_2 * length_of_path + weight_3 * com_cost

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
