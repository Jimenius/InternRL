import os
from collections import deque
import random
import numpy as np
from tensorflow.keras.models import load_model, clone_model
from RLAgents.core import Agent
from utils import TFutils, Torchutils
from utils.utils import Epsilon_Greedy

class GeneralDQNAgent(Agent):
    '''
    Description:
        Deep Q-Learning agent

    Reference:
        Mnih et al, Playing Atari with Deep Reinforcement Learning
        Mnih et al, Human-level Control through Deep Reinforcement Learning
        Hasselt et al, Deep Reinforcement Learning with Double Q-Learning
        Wang et al, Dueling Network Architectures for Deep Reinforcement Learning
    '''

    def __init__(self, capacity = 100, max_step = 0, double = False,
                 epsilon = 1., epsilon_decay_type = 'Exponential', epsilon_decay = 0.9, epsilon_end = 0.01,
                 network = None, batch_size = 32, update = 1, backend = 'Tensorflow', verbose = 1, **kwargs):

        # Initialize parameters
        super(GeneralDQNAgent, self).__init__(**kwargs)
        self.memory = deque(maxlen = capacity)
        self.max_step = max_step
        self.double = double
        self.explore_rate = epsilon
        self.explore_decay = epsilon_decay
        self.explore_decay_type = epsilon_decay_type.upper()
        self.explore_rate_min = epsilon_end
        self.batch_size = batch_size
        if 0 < update < 1:
            self.target_update = update # Soft update
        elif update >= 1:
            self.target_update = int(update) # Hard update
        else:
            raise ValueError('Target update should be greater than 0. (0, 1) for soft update, [1, inf] for hard update.')
        self.backend = backend.upper()
        self.verbose = verbose

        # Initialize the agent
        try:
            self.load_brain(self.brain)
        except:
            network_path = network['PATH']
            if self.backend == 'TENSORFLOW':
                self.QNet = TFutils.ModelBuilder(network_path)
        
        try:
            if self.backend == 'TENSORFLOW':
                self.QTargetNet = clone_model(self.QNet)
                optimizer = TFutils.get_optimizer(name = network['OPTIMIZER'], learning_rate = float(network['LEARNING_RATE']))
                self.QNet.compile(optimizer = optimizer, loss = 'mse')
        except:
            print('Test mode, fail to initialize the network otherwise')

    def _train_net(self):
        batch = random.sample(self.memory, self.batch_size) # Sample batch from memory
        s, a, r, ns, mask = zip(*batch) # Reorganize items
        x = np.array(s)
        ns = np.array(ns)
        if self.double: # Double Q-Learning
            nQ = self.QTargetNet.predict(ns)[np.arange(self.batch_size), np.argmax(self.QNet.predict(ns), axis = -1)] * mask # Q'(s', argmax(Q(s')))
        else:
            nQ = np.amax(self.QTargetNet.predict(ns), axis = 1) * mask # max(Q(s', a'))
        y = self.QNet.predict(x)
        y[np.arange(self.batch_size), a] = self.gamma * nQ + r # Q*(s, a) = r + gamma * max(Q'(s', a'))

        self.QNet.fit(x, y, verbose = self.verbose) # Train network
    
    def learn(self, max_epoch = 0, eval = False, logger = None, plot = False):
        '''
        Description:
            Training method
        
        Inputs:
        max_epoch: int
            Maximum learning epochs
        eval: boolean
            Whether to evaluate agent after a epoch of training
        logger: Logger
            Evaluation logger
        plot: boolean
            Whether to plot a figure after training
        '''

        global_step = 0
        for epoch in range(max_epoch):
            state = self.reset()
            for _ in range(self.max_step):
                q = self.QNet.predict(state[np.newaxis, :]).flatten()
                action = Epsilon_Greedy(q, self.explore_rate)
                next_state, reward, terminal, _ = self.step(action)
                self.memory.append((state, action, reward, next_state, not terminal)) # Append relavant information in the queue
                if terminal:
                    break
                state = next_state # Transit to the next state
                if len(self.memory) >= self.batch_size: # Memory is large enough for training network
                    self._train_net()
                    global_step += 1
                    # Update target network
                    if self.target_update >= 1:
                        # Target hard update
                        if global_step % self.target_update == 0:
                            self.QTargetNet.set_weights(self.QNet.get_weights())
                        else:
                            pass # Placeholder for soft update

            # Evaluating current performance
            if eval:
                _ = self.render(num_episode = 1, vis = False, intv = 0, logger = logger)

            # Shrink exploration rate 
            if self.explore_rate > self.explore_rate_min:
                if self.explore_decay_type == 'EXPONENTIAL':
                    self.explore_rate *= self.explore_decay
                elif self.explore_decay_type == 'LINEAR':
                    self.explore_rate -= self.explore_decay
                else:
                    raise ValueError('Unsupported decay type.')

    def load_brain(self, timestamp):
        if self.backend == 'TENSORFLOW':
            self.QNet = load_model('models/DQN/QNet{}.h5'.format(timestamp))

    def save_brain(self, timestamp):
        os.makedirs('models/DQN', exist_ok = True)
        if self.backend == 'TENSORFLOW':
            self.QNet.save('models/DQN/QNet{}.h5'.format(timestamp))

    def control(self, observation):
        '''
        Description:
            Control method
        '''
        
        return np.argmax(self.QNet.predict(observation[np.newaxis, :]).flatten())