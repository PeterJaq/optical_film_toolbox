from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import copy

from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
#from tf_agents.environments import suite_gymbush
from agents.tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

from common.DataLoader import MaterialLoader
from common.TransferMatrix import OpticalModeling
from common.Config import FilmConfig
from common.utils.FilmLoss import film_loss
from common.utils.FilmTarget import film_target, film_weight
from common.utils.Logger import Logger

class FilmEnvironment(py_environment.PyEnvironment):

    def __init__(self, 
                 config_path,
                 random_init,
                 debug):
        super().__init__()

        if debug:
            self.debug = debug
            self.save_log = True
        else:
            self.debug = False
            self.save_log = False

        self.fmConf = FilmConfig(config_path=config_path)
        self.logger = Logger()

        self.opticalModel = OpticalModeling(Materials = self.fmConf.materials,
                                            WLstep    = self.fmConf.WLstep,
                                            WLrange   = self.fmConf.WLrange)

        self.random_init = random_init

        self.target = film_target(self.fmConf.targets, 
                                  self.fmConf.WLstep,
                                  self.fmConf.WLrange) 
                                  
        self.weight = film_weight(self.fmConf.weights, 
                                  self.fmConf.WLstep,
                                  self.fmConf.WLrange)

        self.round_threshold = self.fmConf.round_threshold
        self.end_threshold   = self.fmConf.end_threshold

        self.init_state = self.fmConf.init_state

        self.action_list = [10, 1, 0.1, -10, -1, -0.1]
        self._state      = copy.copy(self.init_state)

        len_state = len(self._state) - 1

        self.round           = 0
        # self.round_threshold = 100

        self.pre_observation = 9999
        self._action_len      = len(self.action_list) * len_state
        self._observation_len = 1 + len(self._state)

        self._action_spec = array_spec.BoundedArraySpec(
                shape   = (),
                dtype   = np.int64,
                minimum = 0,
                maximum = self._action_len,
                name    = 'action')
            
        self._observation_spec = array_spec.BoundedArraySpec(     
                shape   = (self._observation_len, ),
                dtype   = np.float32,
                minimum = 0,
                maximum = 1000,
                name    = 'observation')

    def _reset(self):
        if self.random_init:
            state_shape = (len(self.init_state))
            # print(state_shape)
            self._state = list(np.random.random(state_shape)*100)
        else:
            self._state = copy.copy(self.init_state)

        # print(self._state)

        self._episode_ended = False
        self.round          = 0
        self.pre_observation = 9999

        self.opticalModel.RunSim(self._state)


        if self.debug:
            print('重新进行搜索')
        if self.save_log:
            self.logger.log_record_csv(self._state)


        # 计算observation
        observation_loss = self.opticalModel.simulation_result
        observation_loss = film_loss(aim          = self.target, 
                                weight       = self.weight,
                                observation  = observation_loss,
                                average      = True,
                                debug        = self.debug,
                                betterfgood  = False)
        # print(observation_loss)

        observation = copy.copy(self._state)
        observation.append(observation_loss)

        # observation = self._state.append(observation_loss)
        # print(self._state, observation_loss)
        # print(observation)

        return ts.restart(observation = np.array(observation, dtype=np.float32))

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def current_time_step(self):
        return self._current_time_step

    def _step(self, action):
        """Apply action and return new time_step."""

        if self._episode_ended:
            return self.reset()

        # 计算action
        action_num     = int(action % len(self.action_list))
        action_layer   = int(action / len(self.action_list))

        self._state[action_layer] += self.action_list[action_num]

        if self.debug:
            print(f'Round {self.round} 薄膜结构: {self._state}, 前序观测: {self.pre_observation}')
        if self.save_log:
            self.logger.log_record_csv(self._state)

        # print(f'仿真的状态:{self._state}')
        self.opticalModel.RunSim(self._state)
        

        # 计算observation
        observation_loss = self.opticalModel.simulation_result
        observation_loss = film_loss(aim          = self.target, 
                                weight       = self.weight,
                                observation  = observation_loss,
                                average      = True,
                                debug        = self.debug,
                                betterfgood  = False)

        observation = copy.copy(self._state)
        observation.append(observation_loss)

        #print(self._state, observation)
        # print(observation)

        # Exit Rule I: Exit Env When Structure Out of Bound.
        error_structure = False
        for s in self._state:
            if s < 0:
                error_structure = True
                self._episode_ended = True

        # Exit Rule II: Exit Env When Film Satisfied Aim Performance.
        reach_performance_threshold = False
        if observation_loss >= self.end_threshold:
            reach_performance_threshold = True
            self._episode_ended = True

        # Exit Rule III: Exit Env When Film not Prove in Threshold Round.
        reach_round_threshold = False
        if self.round >= self.round_threshold:
            reach_round_threshold = True
            self._episode_ended = True
        
        # Exit Term
        if self._episode_ended:
            if error_structure:
                reward = -1
            elif reach_performance_threshold:
                reward = 1
            elif reach_round_threshold:
                reward = -0.1

            if self.debug:
                self.logger.log_record_run([reward, observation_loss])

            return ts.termination(observation = np.array(observation, dtype=np.float32),
                        reward      = reward)
            
        # 更新条件
        elif observation_loss < self.pre_observation:
            reward = (self.pre_observation - observation_loss) * 100
            self.pre_observation = min(observation_loss, self.pre_observation)
            self.round = 0
            
            if self.debug:
                self.logger.log_record_run([reward, observation_loss])

            return ts.transition(observation = np.array(observation, dtype=np.float32),
                                 reward      = reward,
                                 discount    = 1.0)
        else:
            self.round += 1
            reward = -0.01

            if self.debug:
                self.logger.log_record_run([reward, observation_loss])

            return ts.transition(observation = np.array(observation, dtype=np.float32),
                                 reward      = reward,
                                 discount    = 1.0)
