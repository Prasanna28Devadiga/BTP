import math


class AStarPlanner:
    # initaite the class with parameters of image matrix, start position, end position, resolution of matrix [x,y]
    def __init__(self,matrix):
        self.matrix = matrix
        self.min_x = 0
        self.max_x = len(matrix[0])
        self.min_y = 0
        self.max_y = len(matrix)

    # Node -> cell in matrix with fields of point -> [x,y], g,h,f,parent_node
    class Node:
        def __init__(self,point,grid_position,heuristic,parent_node):
            self.point = point
            self.grid_position = grid_position
            self.heuristic = heuristic
            self.cost = self.grid_position + self.heuristic
            self.parent_node = parent_node

        def __str__(self):
            return str(self.point[0])+","+str(self.point[1])+","+str(self.parent_node)

    # verifiying node is in matrix and non-obstacle
    def verifyNode(self,node_point):
        px = node_point//(self.max_x)
        py = node_point%(self.max_x)

        if px < self.min_x :
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        if self.matrix[px][py] != 255:
            return False

        return True

    # calculating the grid position from start node
    def calc_grid_position(self,node_point,):
        return (abs(self.start_index[0]-node_point[0]) + abs(self.start_index[1]-node_point[1]))

    # calculating the heuristic estimated from goal node
    def calc_heuristic(self,node_point):
        return (abs(self.goal_index[0]-node_point[0]) + abs(self.goal_index[1]-node_point[1]))

    # xy_index
    def calc_xy_index(self,node_point):
        return ((node_point[1]*(self.max_x))+node_point[0])

    # total path planning algorithm
    def get_path(self,start_index,goal_index):
        self.start_index = start_index
        self.goal_index = goal_index

        start_node = self.Node(self.start_index,0,self.calc_heuristic(self.start_index),-1)
        goal_node = self.Node(self.goal_index,self.calc_grid_position(self.goal_index),0,-1)

        open_set,closed_set = dict(),dict()

        open_set[self.calc_xy_index(self.start_index)] = start_node

        while 1:
            if len(open_set)==0:
                print("open set is empty")
                break

            # choosing the minimum cost (f) as next node
            current_node_key = min(open_set,key=lambda o:open_set[o].cost)
            current_node = open_set[current_node_key]

            # if current_node is goal_node
            if current_node.point[0] == goal_node.point[0] and current_node.point[1] == goal_node.point[1]:
                print("Find goal")
                goal_node.parent_node = current_node.parent_node
                goal_node.cost = current_node.cost
                break

            del open_set[current_node_key]
            closed_set[current_node_key] = current_node

            motion = [[1,0,1],[0,1,1],[-1,0,1],[0,-1,1],[-1,-1,math.sqrt(2)],[-1,1,math.sqrt(2)],[1,-1,math.sqrt(2)],[1,1,math.sqrt(2)]]

            for i in motion:
                motion_node = self.Node([current_node.point[0]+i[0],current_node.point[1]+i[1]],self.calc_grid_position([current_node.point[0]+i[0],current_node.point[1]+i[1]]),self.calc_heuristic([current_node.point[0]+i[0],current_node.point[1]+i[1]]),current_node_key)
                motion_node.cost += i[2]
                motion_node_key = self.calc_xy_index(motion_node.point)

                if not self.verifyNode(motion_node_key):
                    continue
                if motion_node_key in closed_set:
                    continue
                if motion_node_key not in open_set:
                    open_set[motion_node_key] = motion_node
                else:
                    if open_set[motion_node_key].cost > motion_node.cost:
                        open_set[motion_node_key] = motion_node

        path_points = self.calc_final_path(goal_node,closed_set)

        return path_points

    def calc_final_path(self,goal_node,closed_set):
        path_points = []
        parent_index = goal_node.parent_node

        while parent_index!=-1:
            n=closed_set[parent_index]
            path_points.append(n.point)
            parent_index = n.parent_node

        return path_points

