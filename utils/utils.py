import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser as AP

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

def ParseArguments():
    '''
    Description:
        Parse arguments from Shell scripts.
    
    Outputs:
    argparse.Namespace
        Arguments parsed from Shell scripts
    '''

    parser = AP()

    # Core
    parser.add_argument('--ENV', type = str, default = 'Taxi-v2', help = 'Gym environment')
    parser.add_argument('--AGENT', type = str, default = 'QLearning', help = 'Reinforcement Learning Agent')
    
    # Parameters
    parser.add_argument('--GAMMA', type = float, default = 1, help = 'Discount factor')
    parser.add_argument('--MAX_EPOCH', type = int, default = 1000, help = 'Max Training epochs')
    parser.add_argument('--NUM_EPISODE', type = int, default = 1, help = 'Evaluation episode')
    parser.add_argument('--THETA', type = float, default = 1e-6, help = 'Convergence tolerance')
    parser.add_argument('--LR', type = float, default = 1e-2, help = 'Learning rate')
    parser.add_argument('--EPSILON', type = float, default = 0.1, help = 'Exploration rate')

    # Evaluation
    parser.add_argument('--VIS', action = 'store_true', help = 'Visualize evaluation')
    parser.add_argument('--INTERVAL', type = float, default = 0, help = 'Time interval for render visualization')
    parser.add_argument('--EVAL', action = 'store_true', help = 'Learning evaluation')
    parser.add_argument('--MEMORIZE', action = 'store_true', help = 'Save models')
    parser.add_argument('--LOGDIR', type = str, default = '', help = 'Gym environment')
    parser.add_argument('--PLOT', action = 'store_true', help = 'Plot after learning evaluation')

    opt = parser.parse_args()
    
    return opt

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