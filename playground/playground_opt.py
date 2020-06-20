import os 
import sys 
# root_path = os.path.dirname(os.path.dirname(Path().resolve()))
# if root_path not in sys.path:
    # sys.path.append(root_path)

sys.path.append('/home/peterjaq/Project/optical_film_toolbox')


from simulator.TransferMatrix import OpticalModeling
import numpy as np 
import matplotlib.pyplot as plt 
from common.utils.FilmLoss import film_loss
from common.utils.FilmTarget import *
from sko.GA import GA

Demo = True  # set Demo to True to run an example simulation

def schaffer1(p):
    '''
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    '''
    target_config = {
        'Absorption'  : [[280, 0], [300, 0.5], [1000, 1]],
        'Transmission': [[280, 1], [300, 0.5], [1000, 0]],
        'Reflection'  : [[280, 0], [300, 0], [1000, 0]],
    }

    target = film_target(target_config)

    weight_config = {
        'Absorption'  : [[280, 0], [300, 0.5], [1000, 1]],
        'Transmission': [[280, 1], [300, 0.5], [1000, 0]],
        'Reflection'  : [[280, 0], [300, 0], [1000, 0]],
    }

    weight = film_weight(weight_config)

    Device = [
        "SiO2_Malitson",
        "Zn_Werner",
        "SiO2_Malitson",
        "Zn_Werner",
        "Cu_Johnson"
    ]
    OM = OpticalModeling(Device, 
                        WLrange=(280, 1000))

    OM.RunSim(thickness=p)
    # film_loss(aim, weight, observation, average=False, debug=False, betterfgood=True)
    fl = film_loss(target, weight, OM.simulation_result, average=True, betterfgood=True) 
    return fl

ga1 = GA(func=schaffer1, n_dim=5, size_pop=50, max_iter=10, lb=[0.01, 0.01, 0.01, 0.01, 0.01], ub=[200, 200, 200, 200, 200], precision=1e-7)
print(ga1.run())

import pandas as pd
import matplotlib.pyplot as plt

Y_history = pd.DataFrame(ga1.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()