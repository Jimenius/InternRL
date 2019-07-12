import sys
import os
from time import time
from argparse import ArgumentParser as AP
import yaml

import gym

HOME = os.path.abspath('.')
sys.path.append(HOME)
import RLAgents
from utils.utils import ParseConfig, plot_learn
from utils.log import Logger

def main():
    agent_set = set(('POLICYITERATION', 'VALUEITERATION', 'SARSA', 'QLEARNING'))
    tic = int(time()) # Identify the program process by the timestamp
    
    parser = AP()
    parser.add_argument('--config', type = str, help = 'Gym environment')
    opt = parser.parse_args()

    # Load configuration files as dictionaries.
    with open('configs/default.yaml') as f:
        default = yaml.load(f)
    try:
        with open(opt.config) as f:
            cfg = yaml.load(f)
        ParseConfig(cfg, default)
    except:
        cfg = default
        print('No valid configuration file exists. Use default configurations.')

    # Set configurations
    env_name = cfg['ENV']
    agent_name = cfg['AGENT'].upper()
    params = cfg['PARAMETERS']
    gamma = params['GAMMA']
    theta = float(params['THETA'])
    lr = float(params['LR'])
    eps_start = float(params['EPSILON_START'])
    eps_decay = float(params['EPSILON_DECAY'])
    eps_end = float(params['EPSILON_END'])
    max_epoch = int(float(params['MAX_EPOCH']))
    num_episode = int(float(params['NUM_EPISODE']))
    funcs = cfg['FUNCTIONS']
    vis = funcs['VIS']
    intv = funcs['INTERVAL']
    eval_flag = funcs['EVAL']
    memorize = funcs['MEMORIZE']
    logdir = funcs['LOGDIR']
    plot_flag = funcs['PLOT']

    # Setup program logger
    if logdir and logdir[-1] != '/':
        logdir += '/'
    main_logger = Logger('Main', logdir = logdir, logfile = 'Main{}.log'.format(tic))
    main_logger('Program timestamp: {}'.format(tic))
    main_logger('Argument configurations:')
    main_logger(cfg)

    env = gym.make(env_name)
    main_logger('Environment Description:')
    main_logger('Environment type: {}'.format(env.class_name()))
    main_logger('Observation space: {}'.format(env.observation_space))
    main_logger('Action space: {}'.format(env.action_space))

    if agent_name == 'POLICYITERATION':
        agent = RLAgents.PI.PolicyIterationAgent(env = env, theta = theta, gamma = gamma)
        logfile = 'PILearn{}.log'.format(tic)
    elif agent_name == 'VALUEITERATION':
        agent = RLAgents.VI.ValueIterationAgent(env = env, theta = theta, gamma = gamma)
        logfile = 'VILearn{}.log'.format(tic)
    elif agent_name == 'SARSA':
        agent = RLAgents.SARSA.SARSAAgent(env = env, epsilon = eps_start, lr = lr, gamma = gamma)
        logfile = 'SARSALearn{}.log'.format(tic)
    elif agent_name == 'QLEARNING':
        agent = RLAgents.QL.QLearningAgent(env = env, epsilon = eps_start, lr = lr, gamma = gamma)
        logfile = 'SARSALearn{}.log'.format(tic)
    else:
        main_logger('The agent is not supported at this moment.')

    if agent_name in agent_set:
        learn_logger = Logger('MainLearn', logdir = logdir, logfile = logfile)
        print('Start learning...')
        agent.learn(max_epoch = max_epoch, eval = eval_flag, logger = learn_logger)
        toc = int(time())
        main_logger('Learning takes {}s'.format(toc - tic))
        print('Start evaluation...')
        _ = agent.render(num_episode = num_episode, vis = vis, intv = intv, logger = main_logger)
        if memorize:
            print('Saving learned models...')
            agent.save_brain(tic)
            main_logger('The model is saved.')
        if logdir and eval_flag and plot_flag: # Plot can only be drawn when learning is evaluated and logged.
            main_logger('Plot episodic reward of learning process...')
            plot_learn(logdir + logfile)

    main_logger('Program takes {}s'.format(toc - tic))

if __name__ == '__main__':
    main()