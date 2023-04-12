from PIL import Image
import numpy as np


def get_resized_map(base_map, allowed_loss=0.1, resize_step=0.1):
    base_map = base_map.copy()
    if not (1/resize_step).is_integer():
        raise ValueError("resize_size %f if not proper" % resize_step)
    base_img = Image.fromarray(base_map)
    curr_img = base_img.copy()
    curr_resize = 0
    curr_loss = 1
    while curr_loss > allowed_loss:
        curr_resize += resize_step
        curr_width = round(base_img.size[0] * curr_resize)
        curr_height = round(base_img.size[1] * curr_resize)
        curr_img = base_img.resize((curr_width, curr_height), Image.Resampling.NEAREST)
        expanded_image = curr_img.resize(base_img.size, Image.Resampling.NEAREST)
        curr_map = np.array(expanded_image)
        mismatched_cells = np.sum(curr_map != base_map)
        total_cells = base_img.size[0] * base_img.size[1]
        curr_loss = mismatched_cells / total_cells
    return np.array(curr_img), curr_loss, curr_resize


def get_core_points(path, min_point_separation):
    curr_point = path[0]
    next_point = path[1]
    dr = (next_point[0] - curr_point[0])
    if dr == 0:
        prev_slope = np.inf
    else:
        prev_slope = (next_point[1] - curr_point[1]) / dr

    core_points = list()

    last_point = path[0]
    for i in range(1, path.shape[0] - 1):
        curr_point = path[i]
        next_point = path[i + 1]
        dr = (next_point[0] - curr_point[0])
        if dr == 0:
            curr_slope = np.inf
        else:
            curr_slope = (next_point[1] - curr_point[1]) / dr
        if curr_slope != prev_slope:
            if np.sqrt((last_point[0] - curr_point[0]) ** 2 + (
                    last_point[1] - curr_point[1]) ** 2) >= min_point_separation:
                core_points.append(curr_point)
                last_point = curr_point
        prev_slope = curr_slope

    if len(core_points) == 0:
        core_points.append(path[round(path.shape[0] / 2)])

    return np.array(core_points)


def points_to_particle(points, map_shape):
    points = np.array(points, dtype=np.float64)
    points[:, 0] /= map_shape[1]
    points[:, 1] /= map_shape[0]
    particle = points.reshape(1, -1)[0]
    return particle
