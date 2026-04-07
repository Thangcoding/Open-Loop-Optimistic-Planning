import numpy as np 
import copy 
from collections import deque
from env.grid_world import GridEnv,GridObstacleEnv
from .plan import Planner


class DPPSTree:
    def __init__(self, planner):
        self.planner = planner 

        self.sequence_arm = {}

    def get_plan(self):
        self.planner.plan_path = max([seq for seq in self.sequence_arm if self.sequence_arm[seq][1].done == True], key = lambda x : self.sequence_arm[x][1].count)
    
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
                # greedy rollout 
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
            node, sequence_arm = sequence.popleft() 
            if not node.children:
                value = node.value_sequence()
                self.sequence_arm[tuple(sequence_arm)] = (value, node)
            else:
                for c in node.children:
                    next_node = node.children[c]                                                                                                                                                                        
                    sequence.append([next_node, sequence_arm + [c]])                                   

    def plan(self):
        self.planner.decompose_budgets()               
        self.planner.root = DPPSNode(self.planner)     
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
                best_sequence_arm = max([seq for seq in self.sequence_arm], key = lambda x: self.sequence_arm[x][0]) 

            return_value = self.run(self.planner.env, best_sequence_arm)
            self.planner.logger['episode'].append(episode + 1)
            self.planner.logger['return'].append(return_value)

        self.get_plan()   

class DPPSNode:
    def __init__(self, planner,simulate_reward = 0, parent = None, action = None ):
        self.observation_reward =  []
        self.children = {}
        self.parent = parent 
        self.planner = planner 
        self.count = 0
        self.action = action 

        self.alpha = 3  # concentration parameter in DP 
        self.base_measure = {'mean': 0 , 'sigma': 3} # prior in DP 

        self.probability = np.random.default_rng()
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
    def compute_mean(self):
        ''' 
        Dirichlet Process Posterior Sampling
        '''
        mean_sample = np.array([])
        for i in range(5):
            # posterior sample 
            sample = np.array([])
            num_sample = 20
            for k in range(num_sample):
                if np.random.rand() <= (self.alpha)/(self.alpha + len(self.observation_reward)):
                    sample = np.append(sample,self.probability.normal(loc = self.base_measure['mean'] , scale = self.base_measure['sigma']))
                else:
                    sample = np.append(sample,np.random.choice(self.observation_reward))

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
        self.observation_reward.append(reward)
        self.count += 1 
        self.base_measure['mean'] = sum(self.observation_reward)/len(self.observation_reward) 
        self.mu = self.compute_mean()
        
        self.done = done 

    def expand(self, env):
        actions = self.planner.env.action_space.n
        for action in range(actions):
            simulate_reward = env.simulate(action)
            self.children[action] = type(self)(self.planner,simulate_reward, self, action)
        
        idx = self.planner.leaves.index(self)
        self.planner.leaves = self.planner.leaves[:idx] + list(self.children.values()) + self.planner.leaves[idx + 1:] 
        self.planner.nodes += list(self.children.values())

if __name__ == '__main__':
    config = {'gamma': 0.9, 'horizon': None, 'episode': None, 'budget': 10000}
    env = GridObstacleEnv(width= 4, height= 4)
    planner = Planner(config, env)
    learner = DPPSTree(planner)

    learner.plan()
    print(learner.planner.config['horizon'], learner.planner.config['episode'])
    print(learner.planner.logger)
    print(learner.planner.plan_path)
    learner.planner.simulation()

    # learner.planner.visualize_tree()
