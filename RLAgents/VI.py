import os
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
            self._extract_policy()
        except:
            print('Failed to load pretrained models. Newly initialize the agent')
            self.V = np.zeros(self.state_dim)
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
                transition = self.model.P[state][action]
                tv = 0
                for trans_prob, next_state, reward, _ in transition:
                    tv += trans_prob * (reward + self.gamma * self.V[next_state])
                if tv > newV:
                    newV = tv
                    p = action

            self.policy[state] = p

    def learn(self, max_epoch = 1000, eval = False, logger = None):
        '''
        Description:
            Training method
        
        Inputs:
        max_epoch: int
            Maximum learning epochs
        '''

        for _ in range(max_epoch):
            delta = 0
            for state in range(self.state_dim):
                oldV = self.V[state]
                newV = float('-inf')
                for action in range(self.action_dim):
                    transition = self.model.P[state][action]
                    v = 0
                    for trans_prob, next_state, reward, _ in transition:
                        v += trans_prob * (reward + self.gamma * self.V[next_state])
                    
                    if v > newV:
                        newV = v
                    
                self.V[state] = newV
                delta = max(delta, abs(oldV - newV))
            if delta < self.theta:
                break

            # Evaluating current performance
            if eval:
                self._extract_policy()
                _ = self.render(num_episode = 1, vis = False, intv = 0, logger = logger)
            
        self._extract_policy()

    def load_brain(self, timestamp):
        self.V = np.load('models/VI/V{}.npy'.format(timestamp))

    def save_brain(self, timestamp):
        os.makedirs('models/VI', exist_ok = True)
        np.save('models/VI/V{}.npy'.format(timestamp), self.V)

    def control(self, observation):
        '''
        Description:
            Control method
        '''
        
        return self.policy[observation]