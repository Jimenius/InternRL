import os
import numpy as np
from RLAgents.core import Agent
from utils.utils import Epsilon_Greedy

class SARSAAgent(Agent):
    '''
    Description:
        SARSA agent

    Reference:
        Richard S. Sutton and Andrew G. Barto, Reinforcement Learning An Introduction Second Edition, Chapter 6.4.
    '''

    def __init__(self, epsilon = 1., epsilon_decay_type = 'Exponential', epsilon_decay = 0.99, epsilon_end = 0.01, lr = 1e-2, models = None, **kwargs):
        super(SARSAAgent, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.epsilon_decay_type = epsilon_decay_type.upper()
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.learning_rate = lr

        try:
            self.load_brain(models)
        except:
            self.Q = np.zeros((self.state_dim, self.action_dim))

    def learn(self, max_epoch = 1000, eval = False, logger = None):
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

        for _ in range(max_epoch):
            # Exploring Starts, Reference Chapter 5.3
            isd = self.model.isd.copy()
            self.model.isd = np.ones(self.state_dim) / self.state_dim

            state = self.reset()
            terminal = False
            action = Epsilon_Greedy(value = self.Q[state], e = self.epsilon)

            while not terminal:                
                next_state, reward, terminal, _ = self.step(action)
                next_action = Epsilon_Greedy(self.Q[state], self.epsilon)
                self.Q[state][action] += self.learning_rate * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])
                state = next_state
                action = next_action

            self.model.isd = isd

            # Exploration rate decay
            if self.epsilon > self.epsilon_end:
                if self.epsilon_decay_type == 'EXPONENTIAL':
                    self.epsilon *= self.epsilon_decay
                elif self.epsilon_decay_type == 'LINEAR':
                    self.epsilon -= self.epsilon_decay
                else:
                    raise ValueError('Unsupported decay type.')

            # Evaluating current performance
            if eval:
                _ = self.render(num_episode = 1, vis = False, intv = 0, logger = logger)

    def load_brain(self, timestamp):
        self.Q = np.load('models/SARSA/Q{}.npy'.format(timestamp))

    def save_brain(self, timestamp):
        os.makedirs('models/SARSA', exist_ok = True)
        np.save('models/SARSA/Q{}.npy'.format(timestamp), self.Q)

    def control(self, observation):
        '''
        Description:
            Control method
        '''
        
        return np.argmax(self.Q[observation])