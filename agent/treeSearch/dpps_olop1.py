import numpy as np 
import math 
from env.grid_world import GridEnv
from tool.ucb_function import upper_bound_1, kl_upper_bound 
from .plan import Planner

class DPPSTree:
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
        node = self.planner.root
        return_value = 0 
        for t in range(self.planner.config['horizon']):   

            if node.children == {}:
                node.expand() 
                action = np.random.choice(list(node.children.keys()))
            else:
                action, _ = max([c for c in node.children.items()], key = lambda x : x[1].value_upper)

            observation,reward, done , _ = env.step(action)
            return_value += reward*self.planner.config['gamma']**(t + 1)

            node = node.children[action]
            node.update(reward, done)

            if done:
                break 
        
        node.backup_to_root()
        return return_value

    def plan(self):
        self.planner.decompose_budgets()
        self.planner.root = DPPSNode(self.planner)
        self.planner.leaves.append(self.planner.root)
        self.planner.nodes.append(self.planner.root)

        for episode in range(self.planner.config['episode']):
            
            self.planner.env.reset()
            return_value = self.run(self.planner.env)
            self.planner.logger['episode'].append(episode + 1)
            self.planner.logger['return'].append(return_value)

        self.get_plan() # store path 

class DPPSNode:
    def __init__(self, planner, parent = None ):
        self.observation =  []
        self.children = {}
        self.parent = parent 
        self.planner = planner 
        self.count = 0
 
        self.alpha = 3  # concentration parameter in DP 
        self.base_measure = {'mean': 0 , 'sigma': 3} # prior in DP 
    
        self.probability = np.random.default_rng()
        self.depth = self.parent.depth + 1 if self.parent is not None else 1 
        self.mu = 0
        self.value_upper = self.mu  + (self.planner.config['gamma']**(self.planner.config['horizon']-self.depth + 1))/(1 - self.planner.config['gamma'])
        
        self.done = None 

    def compute_mean(self):
        ''' Dirichlet Process Posterior Sampling
        '''
        mean_sample = np.array([])
        for i in range(5):
            # posterior sample 
            sample = np.array([])
            num_sample = 20
            for k in range(num_sample):
                if np.random.rand() <= (self.alpha)/(self.alpha + len(self.observation)):
                    sample = np.append(sample,self.probability.normal(loc = self.base_measure['mean'] , scale = self.base_measure['sigma']))
                else:
                    sample = np.append(sample,np.random.choice(self.observation))
        
            # using stick-breaking construction process 
            weight = np.array([])
            V = 1 
            for k in range(len(sample)):
                beta = V*self.probability.beta(1,self.alpha + k)
                
                if k == len(sample) - 1:
                    weight = np.append(weight,V)
                else:
                    weight = np.append(weight,beta)

                V -= beta 
            
            mean_sample = np.append(mean_sample,np.dot(sample, weight))
        
        return np.mean(mean_sample)

    def update(self, reward, done):
        self.observation.append(reward)
        self.count += 1 
        self.base_measure['mean'] = sum(self.observation)/len(self.observation) 
        self.mu = self.compute_mean() # + math.sqrt(2*math.log(self.planner.config['episode'])/self.count)
        self.done = done 

    def expand(self):
        actions = self.planner.env.action_space.n
        for action in range(actions):
            self.children[action] = type(self)(self.planner, self)
        
        idx = self.planner.leaves.index(self)
        self.planner.leaves = self.planner.leaves[:idx] + list(self.children.values()) + self.planner.leaves[idx + 1:] 
        self.planner.nodes += list(self.children.values())

    def backup_to_root(self):
        if self.children:
            gamma = self.planner.config['gamma']
            # update value upper 
            self.value_upper = self.mu + gamma* np.max([c.value_upper for c in self.children.values()])
        else:
            # leaf node
            self.value_upper = self.mu 

        if self.parent:
            self.parent.backup_to_root()
        
if __name__ == '__main__':
    config = {'gamma': 0.9, 'horizon': None, 'episode': None, 'budget': 10000}
    env = GridEnv(width= 5, height= 5)
    planner = Planner(config, env)
    learner = DPPSTree(planner)

    learner.plan()
    print(learner.planner.config['horizon'], learner.planner.config['episode'])
    print(learner.planner.logger)
    print(learner.planner.plan_path)
    learner.planner.simulation()
    # learner.planner.visualize_tree()


