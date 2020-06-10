import numpy as np 

def film_target(target_config, WLstep=1, WLrange=[280,1000]):
    target = dict()

    for i in ['Absorption', 'Transmission', 'Reflection']:
        target[i] = cal_target(target_config=target_config[i], WLstep=WLstep, WLrange=WLrange)

    return target


def cal_target(target_config, WLstep, WLrange):
    target = np.zeros(WLrange[1] - WLrange[0])
    
    for idx,t in enumerate(target_config):
        if idx == 0:
            target[WLrange[0]:t[0]-WLrange[0]] == t[1]
        else:
            start_idx                  = target_config[idx - 1][0] - WLrange[0]
            end_idx                    = target_config[idx][0]     - WLrange[0]
            target[start_idx: end_idx] = target_config[idx][1]

    return target 

def film_weight(weight_config, WLstep=1, WLrange=[280, 1200]):
    weight = dict()

    for i in ['Absorption', 'Transmission', 'Reflection']:
        weight[i] = cal_weight(weight_config=weight_config[i], WLstep=WLstep, WLrange=WLrange)

    return weight

def cal_weight(weight_config, WLstep, WLrange):
    weight = np.zeros(WLrange[1] - WLrange[0])
    
    for idx,t in enumerate(weight_config):
        if idx == 0:
            weight[WLrange[0]:t[0]] == t[1]
        else:
            start_idx                 = weight_config[idx - 1][0] - WLrange[0]
            end_idx                   = weight_config[idx][0]     - WLrange[0]
            weight[start_idx:end_idx] = weight_config[idx][1]   

    return weight


if __name__ == "__main__":
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

    #print(target, weight)
