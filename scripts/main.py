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
    env = gym.make(opt.ENV)
    logdir = opt.LOGDIR

    if logdir and logdir[-1] != '/':
        logdir += '/'
    agent_name = opt.AGENT.upper()

    if agent_name == 'POLICYITERATION':
        agent = RLAgents.PI.PolicyIterationAgent(env = env, theta = opt.THETA, gamma = opt.GAMMA)
        learn_logger = Logger('MainLearn', logdir = logdir, logfile = 'PILearn{}.log'.format(tic))
        agent.learn(max_epoch = opt.MAX_EPOCH, eval = opt.EVAL, logger = learn_logger)
        toc = int(time())
        print('Learning takes {}s'.format(toc - tic))
        eval_logger = Logger('MainEval', logdir = logdir, logfile = 'PIEval{}.log'.format(tic))
        _ = agent.render(num_episode = opt.NUM_EPISODE, vis = opt.VIS, intv = opt.INTERVAL, logger = eval_logger)
        if opt.MEMORIZE:
            agent.save_brain(tic)
        if opt.LOGDIR and opt.EVAL and opt.PLOT: # Plot can only be drawn when learning is evaluated and logged.
            plot_learn(logdir + 'PILearn{}.log'.format(tic))
    toc = int(time())
    print('Program takes {}s'.format(toc - tic))