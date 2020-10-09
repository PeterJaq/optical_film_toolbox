import FilmCalu.RunModeling as RunModeling
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class optical_film_env(object):

    def __init__(self):
        
        #self.action = action

        self.libname = "data/Index_Refraction_Zn0.16+SiO2.csv"
        self.wavelength_range = [280, 1500]
        self.posstep = 1.0
        self.WLstep = 2.0
        self.plotWL = [450, 600, 700, 950]
        self.mat_H = "Zn0.16"
        self.mat_L = "SiO2"
        self.mat_surface = "Air"
        self.mat_sub = "Cu"

        self.Device = []
        self.globle_step = 0
        self.max_step = 5000
        self.action_list = [0.01, -0.01]
        self.n_actions = len(self.action_list) * 3
        self.n_features = 5
        self.lowest_abs = 0.7
        self.dead_count = 0
        self.best_abs = 0

    def action_op(self, action):

        """
            action_num: add or reduce the thick of layer. 

                        op1: + 1 nm, op2: + 0.1nm, op3: + 0.01nm
                        op41: - 1 nm, op5: - 0.1nm, op6: - 0.01nm

                        [action List] = [1, 0.1, 0.01, -1, -0.1, -0.01]

            action_layer: the layer add or reduce the thick of layer
                          only the H/L layer could be changed. []  
        """
        action_num = int(action % 2)
        action_layer = int(action / 2) + 1
        #self.action_list = [1, 0.1, 0.01, -1, -0.1, -0.01]
        #print(action, action_layer, self.action_list[action_num])
        self.Device[action_layer][1] = self.Device[action_layer][1] + self.action_list[action_num]

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

    def init_Device(self, layer_num=3, mat_structure="LHL"):
        """
        Device = [
                ("Air", 0),  # layer 0, substrate, thickness doesn't mater                      # layer 1
                ("SiO2", 108.0),  # layer 2
                ("Cr", 9.9),
                ("SiO2", 105.8),
                ("Cu", 200)
         ]
        """
        self.Device = []
        self.dead_count = 0

        self.Device.append([self.mat_surface, 1])
        for _ in mat_structure:
            if _ is "H":
                random_thick = 50
                self.Device.append([self.mat_H, random_thick])
            if _ is "L":
                random_thick = 50
                self.Device.append([self.mat_L, random_thick])
        self.Device.append([self.mat_sub, 200])
        
        absorption = RunModeling.cal_film_abs(Device=self.Device, libname=self.libname,
                                                           WLstep=self.WLstep, posstep=self.posstep,
                                                           plotWL=self.plotWL, wavelength_range=self.wavelength_range)

        self.best_abs = absorption
        
        return self.device_2_observation(self.Device), absorption

    def device_2_observation(self, device):
        observation = []
        for i in device:
            observation.append(i[1])
        #print(observation)

        return np.array(observation)


    def run_simulate(self, action):

        done = 0

        per_abs = RunModeling.cal_film_abs(Device=self.Device, libname=self.libname,
                                                         WLstep=self.WLstep, posstep=self.posstep,
                                                         plotWL=self.plotWL, wavelength_range=self.wavelength_range)

        action_layer, action_num = self.action_op(action)

        mean_abs = RunModeling.cal_film_abs(Device=self.Device, libname=self.libname,
                                                         WLstep=self.WLstep, posstep=self.posstep,
                                                         plotWL=self.plotWL, wavelength_range=self.wavelength_range)

        if mean_abs < self.lowest_abs:
            reward = -1
            done = 1
        elif self.dead_count >= 30:
            done = 1
            reward = 0
        elif mean_abs > 0.95:
            done = 1
            reward = 1
        elif self.device_2_observation(self.Device).min() <= 0:
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

        
        return self.device_2_observation(self.Device), reward, done, mean_abs
