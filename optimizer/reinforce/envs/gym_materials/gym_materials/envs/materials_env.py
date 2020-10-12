import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from numpy.core.fromnumeric import shape
from simulator.TransferMatrix import OpticalModeling
from sko.GA import GA

class MaterialsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.thickness_upper_bound = 200
        self.thickness_lower_bound = 1
        self.layer_num = 4

        self.device = [[0, 0], [0, 0], ([0, 0]), (0, 0)]
        self.thickness = [50, 50, 50, 50]
        self.simulator = OpticalModeling(self.Device, WLrange=(350, 1200))
        self.optimizer = GA(func=self.simulator_func, n_dim=6, size_pop=100, max_iter=500, lb=[1, 10, 10, 10, 10, 10], ub=[2, 200, 200, 200, 200, 200], precision=1e-2)

        self.action_space = spaces.Tuple((spaces.Box(low=-1, high=1, shape=100),
                                          spaces.Box(low=-1, high=1, shape=100),
                                          spaces.Box(low=-1, high=1, shape=100),
                                          spaces.Box(low=-1, high=1, shape=100)))

        self.observation_space = spaces.Tuple((spaces.Box(low=0, high=1, shape=600),
                                                spaces.Box(low=0, high=1, shape=600),
                                                spaces.Box(low=0, high=1, shape=600)))

        self.in_search = True

    def step(self, action):
        self._take_action(action)

        self.status = self.env.step()
        reward = self._get_reward()

    
        ob = self._get_state()
        # if ~self.insearch:
        episode_over = self.in_search != True

        return ob, reward, episode_over, {}

    def reset(self):

        self.device = []
        self.thickness = []
        ...
    def render(self, mode='human'):
        ...
    # def close(self):
    # ...

    def _get_reward(self):
        ...

    def _get_state(self):

        self.simulator.Runsim(thickness=self.thickness)

        return self.simulator.total_Absorption, self.simulator.Transmission, self.simulator.Reflection