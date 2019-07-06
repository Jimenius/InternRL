import sys
import os
from time import time

import gym

HOME = os.path.abspath('.')
sys.path.append(HOME)
import RLAgents
from utils.utils import ParseArguments, plot_learn
from utils.log import Logger

if __name__ == '__main__':

    tic = int(time()) # Identify the program process by the timestamp
    
    opt = ParseArguments()    
    logdir = opt.LOGDIR

    if logdir and logdir[-1] != '/':
        logdir += '/'
    main_logger = Logger('Main', logdir = logdir, logfile = 'Main{}.log'.format(tic))
    main_logger('Program timestamp: {}'.format(tic))
    main_logger('Argument configurations:')
    main_logger(opt)

    env = gym.make(opt.ENV)
    agent_name = opt.AGENT.upper()

    if agent_name == 'POLICYITERATION':
        agent = RLAgents.PI.PolicyIterationAgent(env = env, theta = opt.THETA, gamma = opt.GAMMA)
        learn_logger = Logger('MainLearn', logdir = logdir, logfile = 'PILearn{}.log'.format(tic))
        main_logger('Start learning...')
        agent.learn(max_epoch = opt.MAX_EPOCH, eval = opt.EVAL, logger = learn_logger)
        toc = int(time())
        main_logger('Learning takes {}s'.format(toc - tic))
        main_logger('Start evaluation...')
        _ = agent.render(num_episode = opt.NUM_EPISODE, vis = opt.VIS, intv = opt.INTERVAL, logger = main_logger)
        if opt.MEMORIZE:
            main_logger('Saving learned models...')
            agent.save_brain(tic)
        if opt.LOGDIR and opt.EVAL and opt.PLOT: # Plot can only be drawn when learning is evaluated and logged.
            main_logger('Plot episodic reward of learning process...')
            plot_learn(logdir + 'PILearn{}.log'.format(tic))
    elif agent_name == 'VALUEITERATION':
        agent = RLAgents.VI.ValueIterationAgent(env = env, theta = opt.THETA, gamma = opt.GAMMA)
        learn_logger = Logger('MainLearn', logdir = logdir, logfile = 'VILearn{}.log'.format(tic))
        main_logger('Start learning...')
        agent.learn(max_epoch = opt.MAX_EPOCH, eval = opt.EVAL, logger = learn_logger)
        toc = int(time())
        main_logger('Learning takes {}s'.format(toc - tic))
        main_logger('Start evaluation...')
        _ = agent.render(num_episode = opt.NUM_EPISODE, vis = opt.VIS, intv = opt.INTERVAL, logger = main_logger)
        if opt.MEMORIZE:
            main_logger('Saving learned models...')
            agent.save_brain(tic)
        if opt.LOGDIR and opt.EVAL and opt.PLOT: # Plot can only be drawn when learning is evaluated and logged.
            main_logger('Plot episodic reward of learning process...')
            plot_learn(logdir + 'VILearn{}.log'.format(tic))
    elif agent_name == 'SARSA':
        agent = RLAgents.SARSA.SARSAAgent(env = env, epsilon = opt.EPSILON, lr = opt.LR, gamma = opt.GAMMA)
        learn_logger = Logger('MainLearn', logdir = logdir, logfile = 'SARSALearn{}.log'.format(tic))
        main_logger('Start learning...')
        agent.learn(max_epoch = opt.MAX_EPOCH, eval = opt.EVAL, logger = learn_logger)
        toc = int(time())
        main_logger('Learning takes {}s'.format(toc - tic))
        main_logger('Start evaluation...')
        _ = agent.render(num_episode = opt.NUM_EPISODE, vis = opt.VIS, intv = opt.INTERVAL, logger = main_logger)
        if opt.MEMORIZE:
            main_logger('Saving learned models...')
            agent.save_brain(tic)
        if opt.LOGDIR and opt.EVAL and opt.PLOT: # Plot can only be drawn when learning is evaluated and logged.
            main_logger('Plot episodic reward of learning process...')
            plot_learn(logdir + 'SARSALearn{}.log'.format(tic))
    else:
        main_logger('The agent is not supported at this moment.')

    main_logger('Program takes {}s'.format(toc - tic))