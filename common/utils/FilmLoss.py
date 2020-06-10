import numpy as np 
import pandas as pd 

def film_loss(aim, weight, observation, average=False, debug=False, betterfgood=True):
    # Calculate film loss 
    
    loss_absorbation   = np.mean(weight['Absorption'] * (abs(aim['Absorption'] - observation[0])))
    loss_transimission = np.mean(weight['Transmission'] * (abs(aim['Transmission'] - observation[1])))
    loss_refraction    = np.mean(weight['Reflection'] * (abs(aim['Reflection'] - observation[2])))


    # 检查薄膜状态
    if debug:
        print(f'优化过程中的状态: [吸收]{np.mean(observation[0])}, [投射]{np.mean(observation[1])}, [反射]{np.mean(observation[2])}')
        print(f"优化的目标状态: [吸收]{np.mean(aim['Absorption'])}, [透射]{np.mean(aim['Transmission'])}, [反射]{np.mean(aim['Reflection'])}")
        print(f"film_loss: {np.sum([loss_absorbation, loss_transimission, loss_refraction])}")
        print(f"observation: {1 / np.sum([loss_absorbation, loss_transimission, loss_refraction])}")
        
    if average:
        if betterfgood:
        #print(np.sum([loss_absorbation, loss_transimission, loss_refraction]))
            return 1 / np.sum([loss_absorbation, loss_transimission, loss_refraction])
        else:
            return np.sum([loss_absorbation, loss_transimission, loss_refraction])
    else:
        return loss_absorbation, loss_transimission, loss_refraction

