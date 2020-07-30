import os 
import sys 
# root_path = os.path.dirname(os.path.dirname(Path().resolve()))
# if root_path not in sys.path:
    # sys.path.append(root_path)

sys.path.append('F:\Project\optical_film_toolbox')

from simulator.TransferMatrix import OpticalModeling
import numpy as np 
import matplotlib.pyplot as plt 
from common.utils.FilmLoss import film_loss
from common.utils.FilmTarget import *
from sko.GA import GA

Demo = True  # set Demo to True to run an example simulation

thickness_record = []
performance_record = []

best_performance = 0

def schaffer1(p):
    '''
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    '''

    # print(p)
    target_config = {
        'Absorption'  : [[280, 1], [300, 1], [800, 1]],
        'Transmission': [[280, 0], [300, 0], [800, 0]],
        'Reflection'  : [[280, 0], [300, 0], [800, 0]],
    }

    target = film_target(target_config)

    weight_config = {
        'Absorption'  : [[280, 1], [300, 1], [800, 1]],
        'Transmission': [[280, 0], [300, 0], [800, 0]],
        'Reflection'  : [[280, 0], [300, 0], [800, 0]],
    }

    weight = film_weight(weight_config)

    Device = [                
            "SiO2_Gao",
            "MgF2_Dodge-o",
            "TiO2_Siefke",
            "Si_Aspnes",
            "Ge_Nunley",
            "Cu_Johnson"
        ]
    OM = OpticalModeling(Device, 
                        WLrange=(280, 800))

    OM.RunSim(thickness=p)
    # film_loss(aim, weight, observation, average=False, debug=False, betterfgood=True)
    fl, fa, ft, fr = film_loss(target, weight, OM.simulation_result, average=True, betterfgood=False) 
    # if fa > best_performance:
    #     print(fa)
    #     print(p)
    # print(p)
    thickness_record.append(p)
    performance_record.append([fl, fa, ft, fr])
    return fl


ga1 = GA(func=schaffer1, n_dim=6, size_pop=100, max_iter=500, lb=[1, 10, 10, 10, 10, 10], ub=[2, 200, 200, 200, 200, 200], precision=1e-2)
print(ga1.run())

import pandas as pd
import matplotlib.pyplot as plt

thickness_record_df = pd.DataFrame(thickness_record).to_csv('logs/thickness.csv')
performance_record_df = pd.DataFrame(performance_record).to_csv('logs/performance.csv')

Y_history = pd.DataFrame(ga1.all_history_Y)
print(Y_history.head())
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()