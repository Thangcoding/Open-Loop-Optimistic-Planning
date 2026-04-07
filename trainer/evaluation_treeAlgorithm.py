import numpy as np 
import json 
from env.grid_world import GridEnv, GridObstacleEnv 
from agent.treeSearch.dpps_olop2 import DPPSTree
from agent.treeSearch.klolop2 import KLOLOPTree
from agent.treeSearch.olop import OLOPTree
from agent.treeSearch.plan import Planner


def load_env(env_config, env_name):
    if env_name == 'grid':
        return GridEnv(env_config)
    elif env_name == 'grid_obstacle':
        return GridObstacleEnv(env_config)
    elif env_name == 'high_way':
        pass

def load_agent(agent_config,agent_name):
    planner = Planner(agent_config)
    if agent_name == 'olop':
        return OLOPTree(planner,agent_config)
    elif agent_name == 'klolop':
        return KLOLOPTree(planner,agent_config)
    elif agent_name == 'dpps_olop':
        return DPPSTree(planner,agent_config)

def save_result():
    pass 

def evaluation(agent_config, env_config, lst_agent_eval, lst_env_eval, gamma, budgets, simple_regret, return_value):

    for agent_name in lst_agent_eval:
        for env_name in lst_env_eval:

            logger = {'env':None,
                      'agent':None,
                      'budgets':budgets,
                      'simple_regret':[],
                      'return_value':[],
                      'episode': None,
                      'horizon': None,

            }
            env = load_env(env_config[env_name],env_name)
            agent = load_agent(agent_config,agent_name)
            agent.plan(logger)
            
            
            # store result 
            save_result()

def plot_result():
    pass 
    


if __name__ == '__main__':
    agent_config = {'olop':{'simulated_reward': True,
                            'max_depth': 10
                    },

                    'klolop':{'threshold':"global",
                              'simulate_reward': True,
                              'max_depth':10
                    },

                    'dpps_olop':{'alpha':3,
                                 'prior':'normal',
                                 'mean':0,
                                 'sigma':3,
                                 'simulate_reward': True,
                                 'max_depth':10
                    }
    }

    env_config = {'grid':{'width':10,
                          'height':10,
                          'type_environment': 'DETERMINISTIC',
                          'slip_prob':0.4
                    },
                  'grid_obstacle':{'width':4,
                                  'height':4,
                                  'type_enviroment':'DETERMINISTIC',
                                  'slip_prob':0.4
                  },
                  'high_way':{
                  }
    }
    
    evaluation(
        agent_config,
        env_config,
        lst_agent_eval = ['olop', 'klolop','dpps_olop'],
        lst_env_eval = ['grid','grid_obstacle'],
        gamma = 0.9,
        budgets = 1000,
        simple_regret = True, 
        return_value = True,
        plot = True
    )

                    
    
