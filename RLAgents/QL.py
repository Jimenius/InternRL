import os
import numpy as np
from RLAgents.core import Agent
from utils.utils import Epsilon_Greedy

class QLearningAgent(Agent):
    '''
    Description:
        Q-Learning agent

    Reference:
        Richard S. Sutton and Andrew G. Barto, Reinforcement Learning An Introduction Second Edition, Chapter 6.5.
    '''

    def __init__(self, epsilon = 0.1, lr = 1e-2, **kwargs):
        # Initialize parameters
        super(QLearningAgent, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.learning_rate = lr

        # Initialize the agent
        try:
            self.load_brain(models)
        except:
            self.Q = np.zeros((self.state_dim, self.action_dim))
            #self.Q = np.random.normal(0, 1, size = (self.state_dim, self.action_dim)) # Action values        
    
    def learn(self, max_epoch, eval = False, logger = None, plot = False):
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

        # Exploring Starts, Reference Chapter 5.3
        isd = self.env.env.isd.copy()
        self.env.env.isd = np.ones(self.state_dim) / self.state_dim

        for _ in range(max_epoch):
            state = self.env.reset()
            terminal = False

            while not terminal:
                action = Epsilon_Greedy(value = self.Q[state], e = self.epsilon)
                next_state, reward, terminal, _ = self.env.step(action)
                self.Q[state][action] += self.learning_rate * (reward + self.gamma * np.amax(self.Q[next_state]) - self.Q[state][action])
                state = next_state
            
            if eval:
                _ = self.render(num_episode = 1, vis = False, intv = 0, logger = logger)

        self.env.env.isd = isd

    def load_brain(self, models):
        self.Q = np.load('models/QL/' + models[0] + '.npy')

    def save_brain(self, timestamp):
        os.makedirs('models/QL', exist_ok = True)
        np.save('models/QL/Q{}.npy'.format(timestamp), self.Q)

    def control(self, observation):
        '''
        Description:
            Control method
        '''
        
        return np.argmax(self.Q[observation])