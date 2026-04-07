import numpy 
import random
import copy

class Maze_DFS:
    ''' DFS algorithm purpose for find the path of all node, 
        that mean with find a random spanning tree of a graph cycle.'''
    
    def __init__(self, rows , cols):
        self.rows = rows
        self.cols = cols 

        self.graph = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.maze = [[-1 for _ in range(self.cols*2-1)] for _ in range(self.rows*2-1)]
        self.moves = [(1,0), (0,1), (0,-1),(-1,0)]
        self.start = None 
        self.goal = None 

    def is_valid(self, node):
        ''' check is valid node to traversal '''
        if 0 <= node[0] < self.rows and 0 <= node[1] < self.cols:
            if self.graph[node[0]][node[1]] == None and  node != self.start:
                return True 
        return False 

    def traversal(self, start_node):
        ''' Traversal to find a random spanning tree from cycle graph '''
        stack = [start_node] 
        curr_node = start_node
        space_utilize = self.rows*self.cols - 1 
        while space_utilize > 0: 
        
            neighboors = self.unvisited_node(curr_node)
            if len(neighboors) == 0:
                # all neighboors are visited 
                stack.pop(0)
                if stack == []:
                    return curr_node 
                    break 
                curr_node = stack[0] 
            else:
                next_node = random.choice(neighboors) 
                self.graph[next_node[0]][next_node[1]] = curr_node   
                stack = [next_node] + stack                         
                curr_node = next_node             
                space_utilize -= 1  
        return curr_node 

    def unvisited_node(self,node):
        ''' find all possible neighboors node  '''
        neighboors = []
        for walk in self.moves:
            next_node = (node[0] + walk[0], node[1] + walk[1])
            if self.is_valid(next_node):
                neighboors.append(next_node)
        return neighboors
    
    def generator(self):
        ''' generate maze space by break the location and the wall between each node follow by the path generated '''
        self.start = (random.randint(0, self.rows-1), random.randint(0, self.cols-1))
        self.goal = self.traversal(self.start)
        # mapping node of graph in maze (i, j) --> (i*2 , j*2)
        self.start = (self.start[0]*2 , self.start[1]*2)
        self.goal = (self.goal[0]*2 , self.goal[1]*2)

        # break the wall and location
        for i in range(self.rows):
            for j in range(self.cols):

                # break location 
                if self.graph[i][j] == None: 
                    # that is start node 
                    continue 
                # (i,j) --> (i*2, j*2) :  represent for mapping node from graph in maze 

                # break wall 
                self.maze[i*2][j*2] = 0  

                # break wall 
                path_x , path_y = self.graph[i][j] 
                if  path_x*2 < (self.rows*2 -1) and path_y*2 < (self.cols*2 -1):
                    self.maze[path_x*2][path_y*2] = 0 
                # direction path 
                d_x , d_y = (path_x - i,  path_y - j)
                
                if (i*2 + d_x) < (self.rows*2 -1) and (j*2 + d_y) < (self.cols*2 -1):
                    self.maze[i*2 +d_x][j*2 + d_y] = 0 

        # break location start and goal
        self.maze[self.start[0]][self.start[1]] = 0 
        self.maze[self.goal[0]][self.goal[1]] = 1 

        return self.maze, self.start , self.goal 

class Maze_Prime:
    def __init__(self, rows ,cols):
        self.rows = rows 
        self.cols = cols 

        self.graph = {}
        self.maze = [[-1 for _ in range(self.cols*2 - 1)] for _ in range(self.rows*2 -1)]
        self.lst_node = [(i , j) for j in range(self.cols) for i in range(self.rows)]
        self.start = None 
        self.goal = None 
    def set_up(self):
 
        for i in range(self.rows):
            for j in range(self.cols): 
                node = (i , j)  
                adjacent = [(i + 1, j), (i - 1, j), (i , j + 1), (i , j - 1)] 
                valid_adjacent = [a for a in adjacent if 0 <= a[0] < self.rows and  0 <= a[1] < self.cols]
                self.graph.update({node : {adjacent_node: False for adjacent_node in valid_adjacent}}) 
    
    def is_valid(self, node_1 , node_2): 
        # check created cycle                                                            
        connected_1 = [a for a in self.graph[node_1] if self.graph[node_1][a] == True]   

        if connected_1 == []:
            # not connect before 
            return True 
        stack = connected_1
        start = stack[0]
        visited = [start]         
        while stack != []:  
            start = stack[0]
            if start == node_2: 
                return False  
            adjacents = [a for a in self.graph[start] if self.graph[start][a] == True and a not in visited]
            visited.extend(adjacents) 
            stack.pop(0) 
            stack = adjacents + stack

        return True 

    def conection_node(self):
        number_edge = 0
        while number_edge < self.cols*self.rows-1:
            node = random.choice(self.lst_node)
            lst_adjacent  = [a for a in self.graph[node] if self.graph[node][a] == False]
            if lst_adjacent == []:
                self.lst_node.remove(node)
                continue                            
            adjacent = random.choice(lst_adjacent)  

            if self.is_valid(node , adjacent): 
                # connection edge 
                self.graph[node][adjacent] = True 
                self.graph[adjacent][node] = True 
                number_edge += 1

    def create_cycle(self):

        NUMBER_CYCLE = 14
        count = 0
        while count < NUMBER_CYCLE:
            if self.lst_node == None:
                break 
            node = random.choice(self.lst_node)
            lst_adjacent = [a for a in self.graph[node] if self.graph[node][a] == False]
            if lst_adjacent == []:
                self.lst_node.remove(node)
                continue
            adjacent = random.choice(lst_adjacent)
        
            self.graph[node][adjacent] = True 
            count += 1 
        
    def generator(self, cycle = False):
        self.start , self.goal = random.sample(self.lst_node,2)
        # mapping start and goal on maze 
        self.start = (self.start[0]*2, self.start[1]*2)
        self.goal = (self.goal[0]*2 , self.goal[1]*2)
        
        # set_up edge connection
        self.set_up()
        # generating spanning tree by prime algorithm 
        self.conection_node()
        # create cycle in spanning tree 
        if cycle == True:
            self.create_cycle()

        for i in range(self.rows):
            for j in range(self.cols):
                node = (i , j) 
                # node of graph mapping to maze (i,j) ---> (i*2, j*2)
                # break location
                self.maze[node[0]*2][node[1]*2] = 0 

                for adjacent in self.graph[node]:
                    if self.graph[node][adjacent]:
                        
                        # break wall 
                            # direction path  
                        d_x, d_y = adjacent[0] - node[0],adjacent[1] - node[1]
                        
                        if (node[0]*2 + d_x < self.rows*2 - 1) and (node[1]*2 + d_y < self.cols*2 -1):
                            self.maze[node[0]*2 + d_x][node[1]*2 + d_y] = 0
                        
                        # reset again 
                        self.graph[node][adjacent] = False 
                        self.graph[adjacent][node] = False 
        
        self.maze[self.start[0]][self.start[1]] = 0 
        self.maze[self.goal[0]][self.goal[1]] = 1 

        return self.maze, self.start, self.goal


class Maze_Gen:
    ''' Main generation maze with three algorithm DFS, Prime, and Wilson.
        DFS and Prime work in a tree that purpose to random generate a spanning tree from start point to goal.
        In constract, Wilson work on a graph cycle, then it can creat a maze with more than one path from start point to goal.
        ''' 
    def __init__(self, cols , rows, maze_name = 'Maze_DFS'):
        self.cols = cols 
        self.rows = rows 
        self.maze_name = maze_name
        self.maze_algorithm = {'Maze_DFS':Maze_DFS(self.rows, self.cols),
                                'Maze_Prime': Maze_Prime(self.rows, self.cols),
                                'Maze_Prime_Cycle': Maze_Prime(self.rows, self.cols)}
        self.maze = None
        self.start = None 
        self.goal = None 

    def reset_start(self, node_goal, step = 20):
        ''' fix goal, and traversal to new start location ''' 
        new_start = node_goal
        maze_copy = copy.deepcopy(self.maze)
        for t in range(step):
            # random direction
            valid_dir = []
            for dr in [(0,1),(1,0),(-1,0),(0,-1)]:
                x_new, y_new = new_start[0] + dr[0], new_start[1] + dr[1]
                if 0<= x_new < self.rows*2-1 and 0 <= y_new < self.cols*2 -1 :
                    if maze_copy[x_new][y_new] == 0:
                        valid_dir.append((x_new, y_new))
            if valid_dir == []:
                break 
            maze_copy[new_start[0]][new_start[1]] = -1
            new_start = random.choice(valid_dir)

        return new_start

    def reset_goal(self, node_start, step = 20):
        ''' fix start location, and traversal to new goal ''' 
        new_goal = node_start
        maze_copy = copy.deepcopy(self.maze)
        for t in range(step):
            # random direction
            valid_dir = []
            for dr in [(0,1),(1,0),(-1,0),(0,-1)]:
                x_new, y_new = new_goal[0] + dr[0], new_goal[1] + dr[1]
                if 0<= x_new < self.rows*2-1 and 0 <= y_new < self.cols*2 -1 :
                    if maze_copy[x_new][y_new] == 0:
                        valid_dir.append((x_new, y_new))
            if valid_dir == []:
                break 
            maze_copy[new_goal[0]][new_goal[1]] = -1
            new_goal = random.choice(valid_dir)
        return new_goal
    
    def gen(self):
        if self.maze_name == 'Maze_Prime_Cycle':
            self.maze, self.start , self.goal = self.maze_algorithm[self.maze_name].generator(cycle = True)
        else:
            self.maze, self.start , self.goal = self.maze_algorithm[self.maze_name].generator()


if __name__ == '__main__':
    gen = Maze_Gen(cols = 5, rows= 6, maze_name= 'Maze_Prime')

    gen.gen()

    print(gen.maze)
    print(gen.start)
    print(gen.goal)

        