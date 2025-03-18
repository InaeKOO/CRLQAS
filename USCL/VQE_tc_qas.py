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


from functools import partial


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


if __name__ == "__main__":
    pass







