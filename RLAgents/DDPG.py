import os
from collections import deque
import random
import numpy as np
from tensorflow.keras.models import load_model, clone_model
import tensorflow.keras.backend as K
from RLAgents.core import Agent
from utils import TFutils, Torchutils
from utils.random import OrnsteinUhlenbeckProcess

class DDPGAgent(Agent):
    '''
    Description:
        Deep Deterministic Policy Gradient agent

    Reference:
        Lillicrap el al, Continuous Control with Deep Reinforcement Learning
    '''

    def __init__(self, capacity = 100, max_step = 0,
                 networks = None, batch_size = 32, update = 1, backend = 'Tensorflow', verbose = 1, **kwargs):
        # Initialize parameters
        super(DDPGAgent, self).__init__(**kwargs)
        self.memory = deque(maxlen = capacity)
        self.batch_size = batch_size
        self.target_update = update

        self.max_step = max_step
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
            assert len(networks) == 2
            actor_path = networks[0]['PATH']
            critic_path = networks[1]['PATH']
            if self.backend == 'TENSORFLOW':
                self.Actor = TFutils.ModelBuilder(actor_path)
                self.Critic = TFutils.ModelBuilder(critic_path)
        
        try:
            if self.backend == 'TENSORFLOW':
                self.ActorTarget = clone_model(self.Actor)
                self.CriticTarget = clone_model(self.Critic)
                actor_optimizer = TFutils.get_optimizer(name = networks[0]['OPTIMIZER'], lr = networks[0]['LEARNING_RATE'])
                critic_optimizer = TFutils.get_optimizer(name = networks[1]['OPTIMIZER'], lr = networks[1]['LEARNING_RATE'])
                self.ActorOptimizer = actor_optimizer
                self.Critic.compile(optimizer = critic_optimizer, loss = 'mse')
                self._init_action_train_fn()
        except:
            print('Test mode, fail to initialize the network otherwise')

    def _init_action_train_fn(self):
        state_input = self.Actor.inputs[0]
        action_output = self.Actor.outputs[0]
        q = self.Critic([state_input, action_output])
        updates = self.ActorOptimizer.get_updates(
            params = self.Actor.weights,
            loss = -K.mean(q)
        )
        self.ActorFn = K.function(state_input, q, updates = updates) # Actor training function

    def _train_net(self):
        batch = random.sample(self.memory, self.batch_size) # Sample batch from memory
        s, a, r, ns, mask = zip(*batch)
        x = np.array(s)
        a = np.array(a)
        ns = np.array(ns)
        ta = self.ActorTarget.predict(ns)
        targ = self.CriticTarget.predict([ns, ta]).flatten()
        y = r + self.gamma * targ * mask
        self.Critic.fit([x, a], y, verbose = self.verbose) # Train network
        _ = self.ActorFn(x)
    
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

        process = OrnsteinUhlenbeckProcess(size=self.action_dim[0], theta=.15, mu=0., sigma=.3)
        global_step = 0
        for epoch in range(max_epoch):
            state = self.reset()
            for _ in range(self.max_step):
                noice = process.sample()
                action = self.Actor.predict(state[np.newaxis, :]).flatten() + noice
                next_state, reward, terminal, _ = self.env.step(action)
                self.memory.append((state, action, reward, next_state, not terminal)) # Append relavant information in the queue
                if terminal:
                    break
                state = next_state # Transit to the next state
                if len(self.memory) >= self.batch_size: # Memory is large enough for training network
                    self._train_net()
                    global_step += 1
                    # Update target networks
                    if self.target_update >= 1:
                        # Target hard update
                        if global_step % self.target_update == 0:
                            self.ActorTarget.set_weights(self.Actor.get_weights())
                            self.CriticTarget.set_weights(self.Critic.get_weights())
                        else:
                            pass # Placeholder for soft update

    def load_brain(self, timestamp):
        if self.backend == 'TENSORFLOW':
            self.Actor = load_model('models/DDPG/Actor{}.h5'.format(timestamp))
            self.Critic = load_model('models/DDPG/Critic{}.h5'.format(timestamp))

    def save_brain(self, timestamp):
        os.makedirs('models/DDPG', exist_ok = True)
        if self.backend == 'TENSORFLOW':
            self.Actor.save('models/DDPG/Actor{}.h5'.format(timestamp))
            self.Critic.save('models/DDPG/Critic{}.h5'.format(timestamp))

    def control(self, observation):
        '''
        Description:
            Control method
        '''
        
        return self.Actor.predict(observation[np.newaxis, :]).flatten()