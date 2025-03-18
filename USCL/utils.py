
import configparser
import numpy as np 
import json
from itertools import product

def get_config(config_name,experiment_name, path='configuration_files',
               verbose=True):
    config_dict = {}
    Config = configparser.ConfigParser()
    Config.read('{}/environment/configuration_files/{}{}'.format(path,config_name,experiment_name))
    for sections in Config:
        config_dict[sections] = {}
        for key, val in Config.items(sections):
            try:
                config_dict[sections].update({key: int(val)})
            except ValueError:
                config_dict[sections].update({key: val})
            floats = ['learning_rate',  'dropout', 'alpha', 
                      'beta', 'beta_incr', 'a', 'gamma', 'c', 
                      'maxfev', 'lamda', 'beta_1', 'beta_2',
                      'maxfev1', 'maxfev2', 'maxfev3',
                      "shift_threshold_ball","succes_switch","tolearance_to_thresh","memory_reset_threshold",
                      "fake_min_energy","_true_en","n_shots", "err_mitig", "rand_halt"]
            strings = ['fn_type', 'method','agent_type',
                       "agent_class","init_seed","init_path","init_thresh","method",
                       "mapping","optim_alg", "curriculum_type"]
            lists = ['noise_values','episodes','neurons', 'accept_err','epsilon_decay',"epsilon_min",
                     "epsilon_decay",'final_gamma','memory_clean',
                     'update_target_net', 'epsilon_restart', "thresholds", "switch_episodes"]  
            if key in floats:
                config_dict[sections].update({key: float(val)})
            elif key in strings:
                config_dict[sections].update({key: str(val)})
            elif key in lists:
                config_dict[sections].update({key: json.loads(val)})
    del config_dict['DEFAULT']
    return config_dict


def dictionary_of_actions(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations.
    """
    dictionary = dict()
    i = 0

    for c, x in product(range(num_qubits),
                        range(1, num_qubits)):
        dictionary[i] = [c, x, num_qubits, 0]
        i += 1

    """h  denotes rotation axis. 1, 2, 3 -->  X, Y, Z axes """
    for r, h in product(range(num_qubits),
                        range(1, 4)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1

    valid_actions = []
    for k in dictionary.keys():
        act = dictionary[k]
        ctrl = act[0]
        targ = (act[0] + act[1]) % num_qubits
        tup = (ctrl, targ)

        if tup in [(0, 1), (1, 0), (1, 2), (2, 1), (1, 3), (3, 1), (4, 0)]:
            valid_actions.append(act)

    dictionary_update = dict()
    for val_act_no, val_act in enumerate(valid_actions):
        dictionary_update[len(valid_actions) - 1 - val_act_no] = val_act
    return dictionary_update


def dict_of_actions_revert_q(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations. Systems have reverted order to above dictionary of actions.
    """
    dictionary = dict()
    i = 0

    for c, x in product(range(num_qubits - 1, -1, -1),
                        range(num_qubits - 1, 0, -1)):
        dictionary[i] = [c, x, num_qubits, 0]
        i += 1

    """h  denotes rotation axis. 1, 2, 3 -->  X, Y, Z axes """
    for r, h in product(range(num_qubits - 1, -1, -1),
                        range(1, 4)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1

    valid_actions = []
    for k in dictionary.keys():
        act = dictionary[k]
        ctrl = act[0]
        targ = (act[0] + act[1]) % num_qubits
        tup = (ctrl, targ)
        if tup in [(0, 1), (1, 0), (1, 2), (2, 1), (1, 3), (3, 1), (4, 0)]:
            valid_actions.append(act)

    dictionary_update = dict()
    for val_act_no, val_act in enumerate(valid_actions):
        dictionary_update[len(valid_actions) - 1 - val_act_no] = val_act
    return dictionary_update

if __name__ == '__main__':
    pass
   
