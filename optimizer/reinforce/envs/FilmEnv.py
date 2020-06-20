from tensorforce.environments import Environment

class FilmEnvironment(Environment):

    def __init__(self):
        super().__init__()

    def states(self):
        pass 

    def actions(self):
        pass 

    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    def close(self):
        super().close()

    def reset(self):
        return state 

    def execute(self):
        pass 
        next_state = 0
        terminal = 0
        reward = 0

        return next_state, terminal, reward 