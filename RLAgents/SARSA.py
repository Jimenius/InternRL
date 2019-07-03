import os
import numpy as np
from RLAgents.core import Agent
from utils.utils import Epsilon_Greedy

class SARSAAgent(Agent):
    '''
    Description:
        SARSA agent
    '''

    def __init__(self, epsilon = 0.1, lr = 1e-2, models = None, **kwargs):
        super(SARSAAgent, self).__init__(**kwargs)
        self.epsilon = None
        self.learning_rate = lr

        try:
            self.load_brain(models)
        except:
            self.Q = np.random.normal(0, 1, size = (self.state_dim, self.action_dim)) # Action values

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
            state = self.env.reset()
            terminal = False
            action = Epsilon_Greedy(value = self.Q[state], e = self.epsilon)

            while not terminal:                
                next_state, reward, terminal, _ = self.env.step(action)
                next_action = Epsilon_Greedy(self.Q[state], self.epsilon)
                self.Q[state][action] += self.learning_rate * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])
                state = next_state
                action = next_action

            if eval:
                _ = self.render(num_episode = 1, vis = False, intv = 0, logger = logger)

    def load_brain(self, models):
        self.Q = np.load('models/SARSA/' + models[0] + '.npy')

    def save_brain(self, timestamp):
        os.makedirs('models/SARSA', exist_ok = True)
        np.save('models/SARSA/Q{}.npy'.format(timestamp), self.Q)

    def control(self, observation):
        '''
        Description:
            Control method
        '''
        
        return np.amax(self.Q[observation])