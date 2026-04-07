import numpy as np 
import math 
from collections import deque 
from tool.ucb_function import kl_upper_bound 
from env.grid_world import GridEnv, GridObstacleEnv
from .plan import Planner 

class KLOLOPTree:
    def __init__(self, planner):
        self.planner = planner 
    
        self.sequence_arm = {}

    def get_plan(self):
        self.planner.plan_path = max([seq for seq in self.sequence_arm], key = lambda x : self.sequence_arm[x][1].count)
    
    def run(self, env, best_sequence_arm):
        node = self.planner.root 
        return_value = 0 
        env.reset()
        print(best_sequence_arm)
        for t in range(self.planner.config['horizon']):
            if t  < len(best_sequence_arm):
                action = best_sequence_arm[t]
            else:
                assert not node.children
                node.expand(env)
                action,_ = max(list(node.children.items()), key = lambda x: x[1].mu)

            observation, reward, done, _ = env.step(action)
            return_value += reward*self.planner.config['gamma']**(t + 1)

            node = node.children[action]
            node.update(reward,done)

            if done:
                node.done = True 
                break 

        return return_value 

    def compute_sequence(self):
        '''  Traversal Breath First Search to compute sequence action  '''
        node = self.planner.root 

        sequence = deque([[node.children[c],[c]] for c in node.children])

        while sequence:
            node, sequence_action = sequence.popleft() 
            if not node.children:
                value = node.value_sequence() 
                self.sequence_arm[tuple(sequence_action)] = (value, node)
            else: 
                for c in node.children: 
                    next_node = node.children[c]                                                                                                                                                                        
                    sequence.append([next_node, sequence_action + [c]])                                     

    def plan(self):
        self.planner.decompose_budgets()               
        self.planner.root = KLOLOPNode(self.planner)     
        self.planner.leaves.append(self.planner.root)  
        self.planner.nodes.append(self.planner.root)   

        for episode in range(self.planner.config['episode']):
            if episode == 0:                         
                # null sequenc at the first episode  
                best_sequence_arm = []            
            else:                                      
                # update sequence action value       
                # reset sequence action              
                self.sequence_arm = {}              
                self.compute_sequence()             
                best_sequence_arm, _ = max([seq for seq in self.sequence_arm.items()], key = lambda x: x[1][0]) 
            
            return_value = self.run(self.planner.env, best_sequence_arm)
            self.planner.logger['episode'].append(episode + 1)
            self.planner.logger['return'].append(return_value)

        self.get_plan()

        return 
    
class KLOLOPNode:
    def __init__(self, planner,simulate_reward = 0 , parent = None, action = None ):
        self.children = {}
        self.parent = parent 
        self.planner = planner 
        self.count = 0
        self.cumulative_reward = 0 
        self.action = action

        self.depth = self.parent.depth + 1 if self.parent is not None else 0
        self.mu = simulate_reward

        self.done = False 

    def prefix(self):
        gamma = self.planner.config['gamma']
        return self.parent.prefix() + self.mu*gamma if self.parent is not None else self.mu 
    
    def value_sequence(self):
        gamma , horizon = self.planner.config['gamma'], self.planner.config['horizon']
        if self.done == True:
            return self.prefix() 
        return self.prefix() + (gamma**(self.depth + 1))*(1 - gamma**(horizon - self.depth))/(1 - gamma) if self.children is not None else None   

    def update(self, reward, done):
        
        assert reward <= 1 

        self.cumulative_reward += reward 
        self.count += 1 
        threshold = math.log(self.planner.config['episode'])
        self.mu = kl_upper_bound(self.cumulative_reward, self.count, threshold= threshold) 
 
        self.done = done 

    def expand(self, env):
        actions = self.planner.env.action_space.n
        for action in range(actions):
            simulate_reward= env.simulate(action)
            self.children[action] = type(self)(self.planner,simulate_reward, self, action)
        
        idx = self.planner.leaves.index(self)
        self.planner.leaves = self.planner.leaves[:idx] + list(self.children.values()) + self.planner.leaves[idx + 1:] 
        self.planner.nodes += list(self.children.values())

if __name__ == '__main__':
    config = {'gamma': 0.8, 'horizon': None, 'episode': None, 'budget': 10000}
    env = GridObstacleEnv(width= 4, height= 4)
    planner = Planner(config, env)
    learner = KLOLOPTree(planner)

    learner.plan()
    print(learner.planner.config['horizon'], learner.planner.config['episode'])
    print(learner.planner.logger)
    print(learner.planner.plan_path)
    learner.planner.simulation()
    # print(learner.sequence_arm)
    # learner.planner.visualize_tree()

