from simulator.TransferMatrix import OpticalModeling
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sko.GA import GA
import random


class MaterialEnv(object):

    def __init__(self) -> None:

        self.thickness_upper_bound = 200
        self.thickness_lower_bound = 1
        self.layer_num = 4

        self.device = []
        self.thickness = []
        self.simulator = OpticalModeling(self.Device, WLrange=(350, 1200))
        self.optimizer = GA(func=self.simulator_func, n_dim=6, size_pop=100, max_iter=500, lb=[1, 10, 10, 10, 10, 10], ub=[2, 200, 200, 200, 200, 200], precision=1e-2)

    def simulator_func(self):

        self.simulator.Runsim(thickness=self.thickness)

        return loss 

    def action_op(self, action, action_layer, action_dict):

        # action_num = int(action % len(self.action_list))
        # action_layer = int(action / len(self.thickness)) + 1

        self.device[action_layer] = action_dict[action]

        return action, action_layer


    def init_material(self, layer_num=5):

        self.device = []
        self.thickness = []

        return self.thickness

