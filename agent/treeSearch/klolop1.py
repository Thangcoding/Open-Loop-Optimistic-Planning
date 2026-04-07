import numpy as np 
import math 
from tool.ucb_function import kl_upper_bound 
from env.grid_world import GridEnv
from .plan import Planner 

class KLOLOPTree:
    def __init__(self, planner):
        self.planner = planner 
    
    def get_plan(self):
        sequence_action = []
        node = self.planner.root 
        while node.children != {}:
            action, _ = max([c for c in node.children.items()], key = lambda x : x[1].value_upper)
            sequence_action.append(action)
            node = node.children[action]

        self.planner.plan_path = sequence_action

    def run(self, env):
        env.reset()
        node = self.planner.root 
        return_value = 0
        for t in range(self.planner.config['horizon']):
            if not node.children:
                node.expand()
                action = np.random.choice(list(node.children.keys()))
            else:
                action, value = max([c for c in node.children.items()], key = lambda x : x[1].value_upper) 

            observation , reward , done , _ = env.step(action)
            return_value += reward*self.planner.config['gamma']**(t + 1)
            node = node.children[action]
            node.update(reward)
            
            if done:
                break 
        
        node.backup_to_root()
        return return_value 

    def plan(self):
        self.planner.decompose_budgets()
        self.planner.root = KLOLOPNode(self.planner,None)
        self.planner.leaves.append(self.planner.root)
        self.planner.nodes.append(self.planner.root)

        for episode in range(self.planner.config['episode']):
            return_value = self.run(self.planner.env)
            self.planner.logger['episode'].append(episode + 1)
            self.planner.logger['return'].append(return_value)

        self.get_plan()

class KLOLOPNode:
    def __init__(self, planner ,parent = None):
        self.planner = planner 
        self.children = {}
        self.parent = parent 
        self.mu = 0
        self.cumulative_reward = 0 
        self.count = 0

        self.depth = self.parent.depth + 1 if self.parent is not None else 0
        
        self.value_upper =  self.mu + (self.planner.config['gamma']**(self.depth + 1)/(1 - self.planner.config['gamma'])) 
        self.done = None 

    def update(self, reward):
        
        assert 0 <= reward <= 1 

        self.cumulative_reward += reward 
        self.count += 1 
        threshold = math.log(self.count) + 2*math.log(math.log(self.count) + 1)
        self.mu = kl_upper_bound(self.cumulative_reward, self.count, threshold= threshold) 
    
    def expand(self):
        actions = self.planner.env.action_space.n
        for action in range(actions):
            self.children[action] = type(self)(self.planner, self)
        
        idx = self.planner.leaves.index(self)
        self.planner.leaves = self.planner.leaves[:idx] + list(self.children.values()) + self.planner.leaves[idx+1:]
        self.planner.nodes += list(self.children.values())
    
    def backup_to_root(self): 
        if self.children: 
            gamma = self.planner.config['gamma']
            self.value_upper = self.mu + gamma*np.max([c.value_upper for c in self.children.values()])
        else:
            print(self.depth)
            self.value_upper = self.mu

        if self.parent:
            self.parent.backup_to_root()

if __name__ == '__main__':
    config = {'gamma': 0.9, 'horizon': None, 'episode': None, 'budget': 200}
    env = GridEnv(width= 5, height= 5)
    planner = Planner(config, env)
    learner = KLOLOPTree(planner)

    learner.plan()
    print(learner.planner.config['horizon'], learner.planner.config['episode'])
    print(learner.planner.logger)
    print(learner.planner.plan_path)
    learner.planner.simulation()
    learner.planner.visualize_tree()


