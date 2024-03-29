from time import sleep

class Agent(object):
    '''
    Description:
        General agent super class
    '''

    def __init__(self, env, gamma = 1, brain = None):
        self.env = env
        self.model = env.env # Internal environment model alias
        self.reset = self.env.reset # Reset environment alias
        self.step = self.env.step # Step method alias

        obs = env.observation_space
        acs = env.action_space
        if hasattr(obs, 'n'):
            self.state_dim = obs.n
        else:
            self.state_dim = obs.shape

        if hasattr(acs, 'n'):
            self.action_dim = acs.n
        else:
            self.action_dim = acs.shape
            self.action_range = (acs.low, acs.high)

        # Parameters
        self.gamma = gamma

        # Trained model
        self.brain = brain

    def learn(self):
        raise NotImplementedError # To be completed by subclasses

    def load_brain(self):
        raise NotImplementedError # To be completed by subclasses

    def save_brain(self):
        raise NotImplementedError # To be completed by subclasses
    
    def control(self, observation):
        raise NotImplementedError # To be completed by subclasses

    def render(self, num_episode = 1, vis = False, intv = 1, logger = None):
        '''
        Description:
            Evaluate and visualize the result

        Inputs:
        num_episode: int
            Number of render episodes
        vis: boolean
            Action values
        intv: float
            Time interval for env.render()
        logger: Logger
            logger
            
        Outputs:
        float
            Average cumulative rewards achieved in multiple episodes
        '''
        
        avg_reward = 0 # Average reward of episodes
        for episode in range(num_episode):
            cumulative_reward = 0 # Accumulate rewards of steps in an episode
            terminal = False
            observation = self.env.reset()
            while not terminal:
                if vis:
                    self.env.render()
                    sleep(intv)
                try:
                    action = self.control(observation)
                except NotImplementedError:
                    action = self.env.action_space.sample()
                observation, reward, terminal, _ = self.env.step(action)
                cumulative_reward += reward
                
            avg_reward += cumulative_reward
            if vis:
                self.env.render()
                logtxt = 'Episode {} ends with cumulative reward {}.'.format(episode, cumulative_reward)
                try:
                    logger(logtxt)
                except:
                    print(logtxt)

        if num_episode > 0: # Avoid divided by 0
            avg_reward /= num_episode
            logtxt = 'The agent achieves an average reward of {} in {} episodes.'.format(avg_reward, num_episode)
            try:
                logger(logtxt)
            except:
                print(logtxt)
        self.env.close()
        return avg_reward

class RandomAgent(Agent):
    '''
    Description:
        Random agent as a baseline
    '''

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
        print('Random agent does not need to learn.')
    
    def control(self, observation):
        '''
        Description:
            Control method
        '''
        
        return self.env.action_space.sample()