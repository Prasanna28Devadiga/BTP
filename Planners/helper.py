import numpy as np
def cost_func(path,map,weight_1,weight_2):
    """
    Returns the cost calculated as a weighted sum of the obstacle violations and the length of the path

    Args:
        path (array): array of pixels containing the path 
        map (array): array of pixels containing obstacles
        weight_1 (float): weight given to obstacle avoidance
        weight_2 (float): weight given to shortest length
    """
    violation = 0
    length_of_path = 0
    for path_point in path:
        if map[path_point[0], path_point[1]] == 0:
            violation += 1
    
    for x,y in zip(path,path[1:]):
        length_of_path+= calc_dist(x,y)
    
    cost = weight_1 * violation + weight_2 * length_of_path

    return cost                
    

def calc_dist(p1,p2):
    """
    Returns distance between two points

    Args:
        p1 (tuple): Point 1
        p2 (tuple): Point 2
    """
    dist = np.linalg.norm(p1-p2)
    return dist

def linear_interpolation(start,end,npoints):
    """
    Returns the straight path between start and goal position
    """
    xs, ys = start
    xg, yg = end

    Px = np.linspace(xs, xg, npoints+2)
    Py = np.linspace(ys, yg, npoints+2)

    lin_path = np.concatenate((Px[1:-1], Py[1:-1]))
    return lin_path
