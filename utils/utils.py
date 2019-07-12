import os
import numpy as np
import matplotlib.pyplot as plt

def Epsilon_Greedy(value, e = 0.1):
    '''
    Description:
        Epsilon Greedy policy to select an action based on values
    
    Inputs:
    value: list
        Action values
    e: float
        Exploration rate, Epsilon
    
    Outputs:
    int
        Action index
    '''
    assert len(value.shape) == 1
    assert 0 <= e <= 1

    nA = value.shape[0] # Number of actions
    exploit = np.argmax(value) # Greedy action
    explore = np.random.choice(nA) # Randomly select an action
    a = np.random.choice((exploit, explore), p = (1 - e, e)) # Choose between greedy and random
    return a

def ParseConfig(cfg, default):
    '''
    Description:
        Complete user configurations in-place with default configurations.
    
    Inputs:
    cfg: dict
        User configurations
    default: dict
        Default configurations
    '''

    for k in default:
        if isinstance(default[k], dict):
            ParseConfig(cfg.setdefault(k, {}), default[k]) # Recursively set sub-dictionaries
        else:
            cfg.setdefault(k, default[k])

def plot_learn(logfile):
    '''
    Description:
        Plot learning evaluation.
    
    Inputs:
    logfile: str
        Path to the log file
    '''

    name = logfile.split('/')[-1][:-4]
    with open(logfile, 'r') as f:
        lines = f.readlines()[:-1]
        
    rlist = [] # Average cumulative rewards in each epoch
    for line in lines:
        r = float(line.split()[-4])
        rlist.append(r)
    
    plt.plot(rlist)
    plt.title('Learning Visualization')
    plt.xlabel('Epoch')
    plt.ylabel('Return')
    os.makedirs('imgs/', exist_ok = True)
    plt.savefig('imgs/{}.jpg'.format(name))