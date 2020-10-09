from simulator.TransferMatrix import OpticalModeling
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random


class FilmEnv(object):

    def __init__(self):

        self.Device = [ "SiO2_Malitson",
                        "Ti_Johnson",
                        "SiO2_Malitson",
                        "Ti_Johnson",
                        "Cu_Johnson"]
        
        self.simulator = OpticalModeling(self.Device, WLrange=(350, 1200))

        self.thickness = [100, 100, 100, 100, 200]

        self.globle_step = 0
        self.max_step = 5000
        self.action_list = [1, -1, 0.1, -0.1]
        self.n_actions = len(self.action_list) * len(self.thickness)
        self.n_features = 5
        self.lowest_abs = 0.6
        self.dead_count = 0
        self.best_abs = 0

    def action_op(self, action):

        """
            action_num: add or reduce the thick of layer. 

                        op1: + 1 nm, op2: + 0.1nm, op3: + 0.01nm
                        op4: - 1 nm, op5: - 0.1nm, op6: - 0.01nm

                        [action List] = [1, 0.1, 0.01, -1, -0.1, -0.01]

            action_layer: the layer add or reduce the thick of layer
                          only the H/L layer could be changed. []  
        """
        action_num = int(action % len(self.action_list))
        action_layer = int(action / len(self.thickness)) + 1
        #self.action_list = [1, 0.1, 0.01, -1, -0.1, -0.01]
        print(action, action_layer, self.action_list[action_num])
        self.thickness[action_layer] = self.thickness[action_layer] + self.action_list[action_num]

        return action_num, action_layer
        
    """  
    def load_config(self, config):

        config = configparser.ConfigParser()
        config.read(config, encoding='utf-8')
        self.libname = config.get('optical_film', 'libname')
        self.wavelen_range = config.get('optical_film', 'wavelength_range')
        self.posstep = config.get('optical_film', 'posstep')
        self.WLstep = config.get('optical_film', 'WLstep')
        self.plotWL = config.get('optical_film', 'plotWL')
        self.mat_H = config.get('optical_film', "mat_H")
        self.mat_L = config.get('optical_film', 'mat_L')
        self.mat_surface = config.get('optical_film', 'mat_surface')
        self.mat_sub = config.get('optical_film', 'mat_sub')

        #print("Success load config information:")

        return 0
    """

    def init_thickness(self, layer_num=5, random_init=True, init_thickness_value=50):
        """
        Device = [
                ("Air", 0),  # layer 0, substrate, thickness doesn't mater                      # layer 1
                ("SiO2", 108.0),  # layer 2
                ("Cr", 9.9),
                ("SiO2", 105.8),
                ("Cu", 200)
         ]
        """
        self.thickness = []
        self.dead_count = 0

        for l in range(layer_num):

            if random_init:
                self.thickness.append(random.randint(1, 100))
            else:
                self.thickness.append(init_thickness_value)
        
        self.simulator.RunSim(thickness=self.thickness)

        absorption = np.mean(self.simulator.simulation_result[0])

        self.best_abs = absorption
        
        return self.thickness, absorption

    # def device_2_observation(self, device):
    #     observation = []
    #     for i in device:
    #         observation.append(i[1])
    #     #print(observation)

    #     return np.array(observation)


    def run_simulate(self, action):

        done = 0

        per_abs = np.mean(self.simulator.simulation_result[0])

        action_layer, action_num = self.action_op(action)
        self.simulator.RunSim(thickness=self.thickness)

        mean_abs = np.mean(self.simulator.simulation_result[0])

        if mean_abs < self.lowest_abs:
            reward = -1
            done = 1
        elif self.dead_count >= 30:
            done = 1
            reward = -0.1
        elif mean_abs > 0.95:
            done = 1
            reward = 1
        elif min(self.thickness) <= 0:
            done = 1
            reward = -1
        else:
            reward = (mean_abs - per_abs) * 100

        if mean_abs > self.best_abs:
            self.best_abs = mean_abs
        if reward < 0:
            self.dead_count += 1
        else:
            self.dead_count = 0
        
        return self.thickness, reward, done, mean_abs
