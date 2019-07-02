import os
from time import time
import numpy as np
from RLAgents.core import Agent

class ValueIterationAgent(Agent):
    '''
    Description:
        Policy Iteration agent

    Reference:
        Richard S. Sutton and Andrew G. Barto, Reinforcement Learning An Introduction Second Edition, Chapter 4.4. 
    '''
    
    def __init__(self, theta = 1e-6, models = None, **kwargs):
        # Initialize parameters
        super(ValueIterationAgent, self).__init__(**kwargs)
        self.theta = theta
        
        # Initialize the agent
        try:
            self.load_brain(models)
            self.policy = self._extract_policy()
        except:
            self.V = np.random.normal(0, 1, size = self.state_dim) # State values
            self.policy = np.zeros(self.state_dim, dtype = np.int32) # Current policy

    def _extract_policy(self):
        '''
        Description:
            Policy extraction bound method, extract policy from state values
        '''
        
        for state in range(self.state_dim):
            p = 0
            newV = float('-inf')
            for action in range(self.action_dim):
                transition = self.env.env.P[state][action]
                tv = 0
                for trans_prob, next_state, reward, _ in transition:
                    tv += trans_prob * (reward + self.gamma * self.V[next_state])
                if tv > newV:
                    newV = tv
                    p = action

            self.policy[state] = p
        
        return self.policy

    def learn(self, max_epoch = 1000):
        '''
        Description:
            Training method
        
        Inputs:
        max_epoch: int
            Maximum learning epochs
        '''
        
        for _ in range(max_epoch):
            delta = self.theta
            for state in range(self.state_dim):
                oldV = self.V[state]
                newV = float('-inf')
                for action in range(self.action_dim):
                    transition = self.env.env.P[state][action]
                    v = 0
                    for trans_prob, next_state, reward, _ in transition:
                        v += trans_prob * (reward + self.gamma * self.V[next_state])

                if v > newV:
                    newV = v
                    
                self.V[state] = newV
                delta = max(delta, abs(oldV - newV))
            if delta < self.theta:
                break
            
        _ = self._extract_policy()
        return self.V

    def load_brain(self, models):
        self.V = np.load('models/VI/' + models[0] + '.npy')

    def save_brain(self, tic):
        os.makedirs('models/VI', exist_ok = True)
        np.save('models/VI/V{}.npy'.format(time()), self.V)

    def control(self, observation):
        '''
        Description:
            Control method
        '''
        
        return self.policy[observation]