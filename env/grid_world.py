import numpy as np 
from gymnasium import Env, spaces 
from gymnasium.envs.registration import register 
from gymnasium.utils import seeding 
from tool.maze_generator import Maze_Gen

class GridEnv(Env):

    def __init__(self, width  = 10, height = 10 ):
        self.x = np.array([0,0])
        self.num_actions = 8
        self.grid_width = width 
        self.grid_height = height 
        self.reward_center = [int(width/2), int(height/2)]
        self.reward_radius = int(width/4)
        self.goal = self.reward_center
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low = np.array([0,0]),high = np.array([self.grid_width - 1, self.grid_height - 1]))
        self.type_enviroment = 'STOCHASTIC'
        self.reward_random = np.random.default_rng()

        self.slip_prob = 0.4

        self.start = np.random.choice(self.reward_radius, 2)
        
    def step(self, action):
        if self.type_enviroment == 'STOCHASTIC':
            if np.random.rand() < self.slip_prob: 
                actions = [i for i in range(self.num_actions)]
                actions.remove(action)

        if action == 0: # up 
            self.x[0] += 1  
        elif action == 1: # down 
            self.x[0] -= 1 
        elif action == 2: # right 
            self.x[1] += 1  
        elif action == 3: # left
            self.x[1] -= 1 
        elif action == 4: # top corner left 
            self.x[0] += 1 
            self.x[1] += 1
        elif action == 5: # top corner right 
            self.x[0] += 1
            self.x[1] -= 1
        elif action == 6: # bottum corner left 
            self.x[0] -= 1
            self.x[1] += 1
        elif action == 7: # bottum corner right 
            self.x[0] -= 1
            self.x[1] -= 1

        done = False 
        state = self.x.copy()
        if np.all(self.x == self.start):
            done = True 
        return state , self.reward(), done, {}

    def reward(self, pos):
        return np.clip(1 - 1/self.reward_radius**2 * ((self.reward_center[0] - pos[0])**2
                                                      + (self.reward_center[1] - pos[1])**2),
                       0, 1)

    def simulate(self, action):
        copy_pos = self.x.copy()
        if self.type_enviroment == 'STOCHASTIC':
            if np.random.rand() < self.slip_prob: 
                actions = [i for i in range(self.num_actions)]
                actions.remove(action)

        if action == 0: # up 
            copy_pos[0] += 1  
        elif action == 1: # down 
            copy_pos[0] -= 1 
        elif action == 2: # right
            copy_pos[1] += 1  
        elif action == 3: # left 
            copy_pos[1] -= 1 
        elif action == 4: # top corner left 
            copy_pos[0] += 1 
            copy_pos[1] += 1
        elif action == 5: # top corner right 
            copy_pos[0] += 1
            copy_pos[1] -= 1
        elif action == 6: # bottum corner left 
            copy_pos[0] -= 1
            copy_pos[1] += 1
        elif action == 7: # bottum corner right 
            copy_pos[0] -= 1
            copy_pos[1] -= 1
        
        return self.reward(copy_pos)
    def reset(self):
        self.x = np.array([0, 0])
    

    def render(self):
        grid = np.full((self.grid_width, self.grid_height), '.', dtype=str)

        for i in range(self.grid_width):
            for j in range(self.grid_height):
                # center of reward 
                if (self.reward_radius <= i < self.reward_center[0] + self.reward_radius) and (self.reward_radius <= j <= self.reward_center[1] + self.reward_radius):
                    grid[(i, j)] = 'C'  
        grid[tuple(map(int, self.goal))] = 'G'
        grid[tuple(map(int, self.x))] = 'A'

        return '\n'.join(" ".join(row) for row in grid)
     
class GridObstacleEnv(Env):
    def __init__(self, width=10, height=10):
        super(GridObstacleEnv, self).__init__()

        self.num_actions = 4
        self.grid_width = width 
        self.grid_height = height 

        # Maze generator
        self.gen_maze = Maze_Gen(width, height, maze_name='Maze_Prime')
        self.gen_maze.gen()

        self.reward_random = np.random.default_rng()
        self.raw_space = self.gen_maze.maze
        self.start = self.gen_maze.start
        self.goal = np.array(self.gen_maze.goal)
        self.x = np.array(self.start)

        # Action/observation space
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.grid_width - 1, self.grid_height - 1]),
            dtype=np.int32
        )

        # environment type
        self.type_enviroment = 'DETERMINISTIC'
        self.slip_prob = 0.4 

    def step(self, action):
        if self.type_enviroment == 'STOCHASTIC':
            if np.random.rand() < self.slip_prob:
                actions = [i for i in range(self.num_actions)]
                actions.remove(action)
                action = np.random.choice(actions)

        if action == 0:   # up
            self.x[0] += 1  
        elif action == 1: # down
            self.x[0] -= 1 
        elif action == 2: # right
            self.x[1] += 1  
        elif action == 3: # left
            self.x[1] -= 1 
        elif action == 4: # top corner left 
            self.x[0] += 1 
            self.x[1] += 1
        elif action == 5: # top corner right 
            self.x[0] += 1
            self.x[1] -= 1
        elif action == 6: # bottom corner left 
            self.x[0] -= 1
            self.x[1] += 1
        elif action == 7: # bottom corner right 
            self.x[0] -= 1
            self.x[1] -= 1

        obs, done, reward = self.reward(self.x)
        return obs, reward, done, {}

    def reward(self, pos):
        done = False 
        # Out of bounds
        if not ((0 <= pos[0] < len(self.raw_space)) and (0 <= pos[1] < len(self.raw_space[0]))):
            obs = np.array([-1, -1])  # OUT
            done = True 
            reward = self.reward_random.uniform(0, 0)
            return obs, done, reward

        # Wall
        if self.raw_space[pos[0]][pos[1]] == -1:
            obs = np.array(pos)
            done = True 
            reward = self.reward_random.uniform(0, 0)
        # Path
        elif self.raw_space[pos[0]][pos[1]] == 0:
            obs = np.array(pos)
            reward = self.reward_random.uniform(0, 0.1)
        # Goal
        else:
            obs = np.array(pos)
            done = True 
            print("Reached goal!")
            reward = self.reward_random.uniform(0.9, 1.0)

        return obs, done, reward
    
    def simulate(self, action):
        copy_pos = self.x.copy()
        if self.type_enviroment == 'STOCHASTIC':
            if np.random.rand() < self.slip_prob: 
                actions = [i for i in range(self.num_actions)]
                actions.remove(action)
                action = np.random.choice(actions)

        if action == 0:   # up
            copy_pos[0] += 1  
        elif action == 1: # down
            copy_pos[0] -= 1 
        elif action == 2: # right
            copy_pos[1] += 1  
        elif action == 3: # left
            copy_pos[1] -= 1 
        elif action == 4: # top corner left 
            copy_pos[0] += 1 
            copy_pos[1] += 1 
        elif action == 5: # top corner right 
            copy_pos[0] += 1  
            copy_pos[1] -= 1
        elif action == 6: # bottom corner left 
            copy_pos[0] -= 1
            copy_pos[1] += 1
        elif action == 7: # bottom corner right 
            copy_pos[0] -= 1
            copy_pos[1] -= 1

        _, _, reward = self.reward(copy_pos)
        return reward

    def render(self):
        grid = np.full((len(self.raw_space), len(self.raw_space[0])), '.', dtype=str)

        grid[tuple(map(int, self.goal))] = 'G'

        for i in range(len(self.raw_space)):
            for j in range(len(self.raw_space[0])):
                if self.raw_space[i][j] == -1:
                    grid[(i, j)] = 'X'  

        if (0 <= self.x[0] < len(self.raw_space) and  
            0 <= self.x[1] < len(self.raw_space[0])):
            grid[tuple(map(int, self.x))] = 'A'

        return '\n'.join(" ".join(row) for row in grid)

    def reset(self):
        self.x = np.array(self.start)
        return self.x

    
register(
    id = 'gridenv-v0',
    entry_point= 'env.Discrete:GridEnv'
)
