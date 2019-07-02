from time import sleep

class Agent(object):
    '''
    Description:
        General agent super class
    '''

    def __init__(self, env, gamma = 1):
        self.env = env
        self.state_dim = self.env.observation_space.n # State dimension
        self.action_dim = self.env.action_space.n # Action dimension

        # Parameters
        self.gamma = gamma

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
        
        avg_reward = 0
        for episode in range(num_episode):
            cumulative_reward = 0
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