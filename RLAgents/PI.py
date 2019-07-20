import os
import numpy as np
from RLAgents.core import Agent

class PolicyIterationAgent(Agent):
    '''
    Description:
        Policy Iteration agent

    Reference:
        Richard S. Sutton and Andrew G. Barto, Reinforcement Learning An Introduction Second Edition, Chapter 4.3. 
    '''
    
    def __init__(self, theta = 1e-6, models = None, **kwargs):
        # Initialize parameters
        super(PolicyIterationAgent, self).__init__(**kwargs)
        self.theta = theta
        
        # Initialize the agent
        try:
            self.load_brain(models)
        except:
            self.V = np.zeros(self.state_dim)
            self.policy = np.zeros(self.state_dim, dtype = np.int32) # Current policy
    
    def _PE(self):
        '''
        Description:
            Policy Evaluation bound method

        Reference:
            Richard S. Sutton and Andrew G. Barto, Reinforcement Learning An Introduction Second Edition, Chapter 4.1. 
        '''

        delta = self.theta
        while delta >= self.theta:
            delta = 0
            for state in range(self.state_dim):
                oldV = self.V[state]
                newV = 0
                transition = self.model.P[state][self.policy[state]]
                for trans_prob, next_state, reward, _ in transition:
                    newV += trans_prob * (reward + self.gamma * self.V[next_state])
                self.V[state] = newV # Asynchronous update
                delta = max(delta, abs(oldV - newV))

        return self.V

    def _PI(self):
        '''
        Description:
            Policy Improvement bound method
                
        Reference:
            Richard S. Sutton and Andrew G. Barto, Reinforcement Learning An Introduction Second Edition, Chapter 4.2. 
        '''
    
        policy_stable = True
        for state in range(self.state_dim):
            oldP = self.policy[state]
            q = np.zeros(self.action_dim)
            for action in range(self.action_dim):
                transition = self.model.P[state][action]
                for trans_prob, next_state, reward, _ in transition:
                    q[action] += trans_prob * (reward + self.gamma * self.V[next_state])
                    
            self.policy[state] = np.argmax(q)
            if oldP != self.policy[state]:
                policy_stable = False
        
        return policy_stable

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
        
        stable = False # Whether the policy is stable
        epoch = 0
        while not stable and epoch < max_epoch:
            _ = self._PE() # Call Policy evaluation bound method
            stable = self._PI() # Call Policy Improvement bound method

            # Evaluating current performance
            if eval:
                _ = self.render(num_episode = 1, vis = False, intv = 0, logger = logger)

            epoch += 1

    def load_brain(self, models):
        self.policy = np.load('models/PI/' + models[0] + '.npy')
        self.V = np.load('models/PI/' + models[1] + '.npy')

    def save_brain(self, timestamp):
        os.makedirs('models/PI', exist_ok = True)
        np.save('models/PI/Policy{}.npy'.format(timestamp), self.policy)
        np.save('models/PI/V{}.npy'.format(timestamp), self.V)

    def control(self, observation):
        '''
        Description:
            Control method
        '''
        
        return self.policy[observation]