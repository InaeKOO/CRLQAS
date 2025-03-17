import tensorcircuit as tc
import jax
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Tuple, Optional, Dict
from scipy.optimize import OptimizeResult
import pickle
import time
import chex

import os

import torch


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  


from functools import partial

from jax.config import config

from jax import numpy as jnp

from environment.VQE_files.quant.quant_lib_jax import state_initializer

from environment.VQE_files.quant.quant_lib_4q_noiseless import H_4_ptm, rho0_4_ptm

from environment.VQE_files.quant.quant_lib_3q_10max import H_3_ptm, rho0_3_ptm

from environment.VQE_files.quant.quant_lib_2q_noiseless import H_2_ptm, rho0_2_ptm


from functools import partial

from environment.VQE_files.VQE_main_4q_noiseless import get_exp_val_4q, get_noiseless_exp_val_4q, get_exp_val_static_4q, get_noiseless_exp_val_static_4q
from environment.VQE_files.VQE_main_4q_noiseless import get_instrs_4q

from environment.VQE_files.VQE_main_3q_10max import get_exp_val_3q, get_noiseless_exp_val_3q, get_exp_val_static_3q, get_noiseless_exp_val_static_3q
from environment.VQE_files.VQE_main_3q_10max import get_instrs_3q

from environment.VQE_files.VQE_main_2q_noiseless import get_exp_val_2q, get_noiseless_exp_val_2q, get_exp_val_static_2q, get_noiseless_exp_val_static_2q
from environment.VQE_files.VQE_main_2q_noiseless import get_instrs_2q


def get_instrs( state, n_qubits =4 ):
    if n_qubits == 4:
        return get_instrs_4q(state)
    elif n_qubits == 3:
        return get_instrs_3q(state)
    elif n_qubits == 2:
        return get_instrs_2q(state)
def ptm_initializer( n_qubits = 4):
    if n_qubits == 4:
        return rho0_4_ptm
    elif n_qubits == 3:
        return rho0_3_ptm
    elif n_qubits == 2:
        return rho0_2_ptm

def get_Hamil_ptm( n_qubits = 4):
    if n_qubits == 4:
        return H_4_ptm
    elif n_qubits == 3:
        return H_3_ptm
    elif n_qubits == 2:
        return H_2_ptm



class Parametric_Circuit:
    def __init__(self,state, Hamil,energy_shift, weights,noise_values = [],
                   Nshots = 1e7):
        self.num_layers = state.shape[0] 
        self.num_qubits = state.shape[2]
        self.st = state_initializer(self.num_qubits)
        self.Nshots = Nshots
        self.noise_values = noise_values
        self.Hamil = Hamil
        if Nshots > 0:
            self.sigma = (Nshots)**(-0.5)
        else:
            self.sigma = 0
        self.energy_shift = energy_shift
        self.weights = np.array(weights)

        if len(self.noise_values) >0:
            self.rho_ptm = ptm_initializer(self.num_qubits)
            self.Hamil_ptm = get_Hamil_ptm( self.num_qubits)

        circ_instrs_np, rt_ps_np = get_instrs(state, self.num_qubits)
        self.circ_instrs_np = circ_instrs_np
        self.rt_ps_np    = rt_ps_np
        self.num_gates   = circ_instrs_np.shape[0]

    def putangles(self,angles):
        circ_instrs_np, circ_instrs_jnp = putangles(angles, self.circ_instrs_np, self.rt_ps_np)
        return circ_instrs_jnp






def putangles(angles, circ_instrs, rpos):
    if len(rpos) > 0:
        circ_instrs[rpos, 1] = angles
    return circ_instrs, jnp.array(circ_instrs)




def get_exp_val_static(angles, param_circ):
    if param_circ.num_qubits == 4:
        return get_exp_val_static_4q(angles, param_circ)
    elif param_circ.num_qubits == 3:
        return get_exp_val_static_3q(angles, param_circ)
    elif param_circ.num_qubits == 2:
        return get_exp_val_static_2q(angles, param_circ)

def get_noiseless_exp_static_val(angles, param_circ):
    if param_circ.num_qubits == 4:
        return get_noiseless_exp_val_static_4q(angles, param_circ)
    elif param_circ.num_qubits == 3:
        return get_noiseless_exp_val_static_3q(angles, param_circ)
    elif param_circ.num_qubits == 2:
        return get_noiseless_exp_val_static_2q(angles, param_circ)

def get_exp_val(angles, param_circ, sigma=0):
    if param_circ.num_qubits == 4:
        return get_exp_val_4q(angles, param_circ, sigma)
    elif param_circ.num_qubits == 3:
        return get_exp_val_3q(angles, param_circ, sigma)
    elif param_circ.num_qubits == 2:
        return get_exp_val_2q(angles, param_circ, sigma)

def get_noiseless_exp_val(angles, param_circ):
    if param_circ.num_qubits == 4:
        return get_noiseless_exp_val_4q(angles, param_circ)
    elif param_circ.num_qubits == 3:
        return get_noiseless_exp_val_3q(angles, param_circ)
    elif param_circ.num_qubits == 2:
        return get_noiseless_exp_val_2q(angles, param_circ)

#



#
#







if __name__ == "__main__":
    pass







