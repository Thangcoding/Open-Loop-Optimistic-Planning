import numpy as np 


random_reward = np.random.default_rng()


ENV = {'1':[2,0.5], '2': [4,3],'3':[1.5,1],'4': [2,2]}

class DPPSArm:
    def __init__(self, arm_id):
        
        self.arm_id = arm_id
        self.observation = []
        self.mu = 0 
        self.count = 0 

        self.alpha = 3 
        self.base_measure = {'mean': 0, 'sigma': 3}

    def sampling(self):
        ''' 
        Dirichlet Process Posterior Sampling
        '''
        mean_sample = np.array([])
        for i in range(5):
            # posterior sample 
            sample = np.array([])
            num_sample = 20
            for k in range(num_sample):
                if np.random.rand() <= (self.alpha)/(self.alpha + len(self.observation)):
                    sample = np.append(sample,random_reward.normal(loc = self.base_measure['mean'] , scale = self.base_measure['sigma']))
                else:
                    sample = np.append(sample,np.random.choice(self.observation))

            # using stick-breaking construction process 
            weight = np.array([])
            V = 1 
            for k in range(len(sample)):
                beta = V*random_reward.beta(1,self.alpha + k)

                if k == len(sample) - 1:
                    weight = np.append(weight,V)
                else:
                    weight = np.append(weight,beta)

                V -= beta 
            
            mean_sample = np.append(mean_sample,np.dot(sample, weight))
        
        return np.mean(mean_sample)            

    def update(self,reward):                   
        self.observation.append(reward)         
        self.count += 1                        
        if self.count == 1:                    
            self.base_measure['mean'] = reward 
        self.mu = self.sampling()
    def play(self, env):
        mean, sigma = env[self.arm_id]

        reward = random_reward.normal(loc = mean,scale= sigma) 
        self.update(reward)  
        return reward            

if __name__ == '__main__':
    lst_arm = {}
    policy = lambda x : max([a for a in x], key = lambda t: t.mu).arm_id
    cumulative_reward = 0 
    history = []
    for i in range(1,5):
        arm = DPPSArm(arm_id = str(i))
        reward = arm.play(env= ENV)
        lst_arm[str(i)] = arm 
    
    for i in range(500):
        arm_id = policy(lst_arm.values())
        history.append(arm_id)
        reward = lst_arm[arm_id].play(env = ENV)    
        cumulative_reward += reward

    print(cumulative_reward)
    print(history)
        

    
