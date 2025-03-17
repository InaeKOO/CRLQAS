import time
from genericpath import exists
from multiprocessing.resource_sharer import stop
from unittest import skip
from collections import Counter
import jax
import jax.numpy as jnp
import time as clock_time
import timeit
import torch
from qulacs import QuantumCircuit
from qulacs.gate import CNOT, RX, RY, RZ
from utils import *
from sys import stdout
from itertools import product
import scipy
import environment.VQE_files.VQE_tc_4q_median as vc_tc
import environment.VQE_files.VQE as vc

import os
import numpy as np
import random
import copy
import environment.curricula_restart_experiment as curricula
from functools import partial
from scipy.optimize import minimize

import environment.VQE_files.VQE_helper as vc_h

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  

try:
    from qulacs import QuantumStateGpu as QuantumState
except ImportError:
    from qulacs import QuantumState

from qulacs import QuantumCircuit, ParametricQuantumCircuit

from typing import List, Callable, Tuple, Optional, Dict
import copy


from scipy.optimize import OptimizeResult




class CircuitEnv():

    def __init__(self, conf, device, args):

        self.random_halt = int(conf['env']['rand_halt'])

        self.num_qubits = conf['env']['num_qubits']
        print(f"num qubits is {self.num_qubits}")

        self.num_layers = conf['env']['num_layers']

        self.n_shots = conf['env']['n_shots']  
        noise_values = list(filter(None, conf['env']['noise_values']))  
        noise_models = ['depolarizing', 'two_depolarizing', 'amplitude_damping']
        self.noise_values = noise_values
        self.noise_models = noise_models[0:len(self.noise_values)]
        if len(self.noise_values) > 0:
            self.phys_noise = True
        else:
            self.phys_noise = False
        self.err_mitig = conf['env']['err_mitig']

        self.ham_mapping = conf['problem']['mapping']
        #

        #

        self.geometry = conf['problem']['geometry'].replace(" ", "_")

        self.fake_min_energy = conf['env']['fake_min_energy'] if "fake_min_energy" in conf['env'].keys() else None
        self.fn_type = conf['env']['fn_type']

        if "cnot_rwd_weight" in conf['env'].keys():
            self.cnot_rwd_weight = conf['env']['cnot_rwd_weight']
        else:
            self.cnot_rwd_weight = 1.


        self.nmc = 500
        self.time = clock_time.time()

        self.noise_flag = True

        self.state_with_angles = conf['agent']['angles']
        self.current_number_of_cnots = 0

        self.problem = conf['problem']['ham_type']

        self.curriculum_dict = {}
        __ham = np.load(f"mol_data/{self.problem}_{self.num_qubits}q_geom_{self.geometry}_{self.ham_mapping}.npz")
        hamiltonian, weights, eigvals, energy_shift = __ham['hamiltonian'], __ham['weights'], __ham['eigvals'], __ham[
            'energy_shift']

        min_eig = conf['env']['fake_min_energy'] if "fake_min_energy" in conf['env'].keys() else min(
            eigvals) + energy_shift

        self.hamiltonian, self.weights, eigvals, self.energy_shift = __ham['hamiltonian'], __ham['weights'], __ham[
            'eigvals'], __ham['energy_shift']

        self.min_eig = self.fake_min_energy if self.fake_min_energy is not None else min(eigvals) + self.energy_shift

        self.min_eig_true = min(eigvals) + self.energy_shift

        self.max_eig = max(eigvals) + self.energy_shift

        self.curriculum_type = conf['env']['curriculum_type']


        if self.curriculum_type == 'MovingThreshold' and conf['agent']['init_net'] and conf['agent']['summary_file']:
            results_path = "results/"
            PATH = f"{results_path}{conf['agent']['summary_file']}{args.seed}"
            print(PATH)
            init_data = np.load(PATH + f".npy", allow_pickle=True).ravel()[0]
            summary_episodes = len(init_data['train'])
            subtraction_idx = int(summary_episodes % 50)


            data = dict(list(init_data['train'].items())[:-subtraction_idx])

            summary_noisy_errors = [(data[ep_no]['errors'][-1]) for ep_no in
                                    range(len(data))]  
            min_summary_done_threshold = data[len(data) - 1]['done_threshold']

            min_summary_noisy_energy = np.min(summary_noisy_errors) + self.min_eig


        self.curriculum_dict[self.geometry[-3:]] = curricula.__dict__[conf['env']['curriculum_type']](conf['env'],
                                                                                                      target_energy=min_eig,
                                                                                                      accept_err=min_summary_done_threshold,
                                                                                                      min_energy_noisy_error=min_summary_noisy_energy)


        self.device = device

        stdout.flush()
        self.state_size = self.num_layers * self.num_qubits * (self.num_qubits + 3 + 3)
        self.step_counter = -1
        self.prev_energy = None
        self.moments = [0] * self.num_qubits
        self.illegal_actions = [[]] * self.num_qubits
        self.energy = 0

        self.action_size = (self.num_qubits * (self.num_qubits + 2))
        self.previous_action = [0, 0, 0, 0]

        if 'non_local_opt' in conf.keys():
            self.global_iters = conf['non_local_opt']['global_iters']
            self.optim_method = conf['non_local_opt']["method"]
            self.optim_alg = conf['non_local_opt']['optim_alg']

            if 'a' in conf['non_local_opt'].keys():
                self.options = {'a': conf['non_local_opt']["a"], 'alpha': conf['non_local_opt']["alpha"],
                                'c': conf['non_local_opt']["c"], 'gamma': conf['non_local_opt']["gamma"],
                                'beta_1': conf['non_local_opt']["beta_1"],
                                'beta_2': conf['non_local_opt']["beta_2"]}

            if 'lamda' in conf['non_local_opt'].keys():
                self.options['lamda'] = conf['non_local_opt']["lamda"]

            if 'maxfev' in conf['non_local_opt'].keys():
                self.maxfev = int(conf['non_local_opt']["maxfev"])

            if 'maxfev1' in conf['non_local_opt'].keys():
                self.maxfevs = {}
                self.maxfevs['maxfev1'] = int(conf['non_local_opt']["maxfev1"])
                self.maxfevs['maxfev2'] = int(conf['non_local_opt']["maxfev2"])
                self.maxfevs['maxfev3'] = int(conf['non_local_opt']["maxfev3"])
        else:
            self.global_iters = 0
            self.optim_method = None


    def step(self, action, train_flag=True):

        """
        Action is performed on the first empty layer.

        Variable 'step_counter' points last non-empty layer.
        """

        next_state = self.state.clone()
        self.step_counter += 1

        """
        First two elements of the 'action' vector describes position of the CNOT gate.
        Position of rotation gate and its axis are described by action[2] and action[3].
        When action[0] == num_qubits, then there is no CNOT gate.
        When action[2] == num_qubits, then there is no Rotation gate.
        """

        ctrl = action[0]
        targ = (action[0] + action[1]) % self.num_qubits
        rot_qubit = action[2]
        rot_axis = action[3]


        self.action = action

        if rot_qubit < self.num_qubits:
            gate_tensor = self.moments[rot_qubit]
        elif ctrl < self.num_qubits:
            gate_tensor = max(self.moments[ctrl], self.moments[targ])

        if ctrl < self.num_qubits:
            next_state[gate_tensor][targ][ctrl] = 1
        elif rot_qubit < self.num_qubits:
            next_state[gate_tensor][self.num_qubits + rot_axis - 1][rot_qubit] = 1

        if rot_qubit < self.num_qubits:
            self.moments[rot_qubit] += 1
        elif ctrl < self.num_qubits:
            max_of_two_moments = max(self.moments[ctrl], self.moments[targ])
            self.moments[ctrl] = max_of_two_moments + 1
            self.moments[targ] = max_of_two_moments + 1

        self.current_action = action
        self.illegal_action_new()



        state = self.state.clone()

        thetas = state[:, self.num_qubits + 3:]
        rot_pos = (state[:, self.num_qubits: self.num_qubits + 3] == 1).nonzero(as_tuple=True)
        angles = thetas[rot_pos]


        self.param_circ = vc_tc.Parametric_Circuit(state,
                                                   jnp.array(self.hamiltonian),
                                                   self.energy_shift, self.weights,
                                                   self.noise_values, Nshots=self.n_shots)



        x0_flag = False

        if self.optim_method in ["SPSA"]:
            x0 = self.adam_spsa_v2(angles)
            x0_flag = True
        elif self.optim_method in ["SPSA3"]:
            x0 = self.adam_spsa_3((angles))
            x0_flag = True
        else:
            x0 = self.cobyla_min(angles)
            x0_flag = True

        energy, energy_noiseless = self.get_energy(x0)

        if x0_flag:
            self.x = x0.__array__()
        else:
            self.x = None
        thetas[rot_pos] = torch.tensor(x0.__array__(), dtype=torch.float)

        for i in range(self.num_layers):
            for j in range(3):
                next_state[i][self.num_qubits + 3 + j, :] = thetas[i][j, :]

        self.state = next_state.clone()
        cnots = self.state[:, :self.num_qubits]
        self.current_number_of_cnots = np.count_nonzero(cnots)

        if self.noise_flag == False:
            energy = energy_noiseless

        self.energy = energy

        #



        if energy < self.curriculum.lowest_energy and train_flag:
            self.curriculum.lowest_energy = copy.copy(energy)

        self.error = float(abs(self.min_eig - energy))
        self.error_noiseless = float(abs(self.min_eig_true - energy_noiseless))



        rwd = self.reward_fn(energy)
        self.prev_energy = np.copy(energy)

        energy_done = int(self.error < self.done_threshold)
        layers_done = self.step_counter == (self.num_layers - 1)
        done = int(energy_done or layers_done)

        self.previous_action = copy.deepcopy(action)


        if self.random_halt:
            if self.step_counter == self.halting_step:
                print(f"Last action of the episode now.")
                print(f"Last angle is {self.x}")
                print(f"noiseless error is {self.error_noiseless}", flush=True)
                done = 1

        if done:
            self.curriculum.update_threshold(energy_done=energy_done)
            self.done_threshold = self.curriculum.get_current_threshold()
            self.curriculum_dict[str(self.current_bond_distance)] = copy.deepcopy(self.curriculum)

        if self.state_with_angles:
            return next_state.view(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done
        else:
            next_state = next_state[:, :self.num_qubits + 3]
            return next_state.reshape(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32,
                                                                        device=self.device), done

    def reset(self):
        """
        Returns randomly initialized state of environment.
        State is a torch Tensor of size (5 x number of layers)
        1st row [0, num of qubits-1] - denotes qubit with control gate in each layer
        2nd row [0, num of qubits-1] - denotes qubit with not gate in each layer
        3rd, 4th & 5th row - rotation qubit, rotation axis, angle
        !!! When some position in 1st or 3rd row has value 'num_qubits',
            then this means empty slot, gate does not exist (we do not
            append it in circuit creator)
        """
        state = torch.zeros((self.num_layers, self.num_qubits + 3 + 3, self.num_qubits))
        self.state = state


        if self.random_halt:
            statistics_generated = np.clip(np.random.negative_binomial(n=40, p=0.6, size=100), 0, 40)
            c = Counter(statistics_generated)
            self.halting_step = c.most_common(1)[0][0]

        self.current_number_of_cnots = 0
        self.current_action = [self.num_qubits] * 4
        self.illegal_actions = [[]] * self.num_qubits


        self.step_counter = -1

        self.moments = [0] * self.num_qubits
        self.current_bond_distance = self.geometry[-3:]
        self.curriculum = copy.deepcopy(self.curriculum_dict[str(self.current_bond_distance)])
        self.done_threshold = copy.deepcopy(self.curriculum.get_current_threshold())

        self.geometry = self.geometry[:-3] + str(self.current_bond_distance)

        __ham = np.load(f"mol_data/{self.problem}_{self.num_qubits}q_geom_{self.geometry}_{self.ham_mapping}.npz")
        self.hamiltonian, self.weights, eigvals, self.energy_shift = __ham['hamiltonian'], __ham['weights'], __ham[
            'eigvals'], __ham['energy_shift']

        self.min_eig = self.fake_min_energy if self.fake_min_energy is not None else min(eigvals) + self.energy_shift
        self.max_eig = max(eigvals) + self.energy_shift

        state = self.state.clone()
        thetas = state[:, self.num_qubits + 3:]
        rot_pos = (state[:, self.num_qubits: self.num_qubits + 3] == 1).nonzero(as_tuple=True)
        angles = thetas[rot_pos]
        x0 = jnp.array(angles, dtype=jnp.float32)


        self.param_circ = vc_tc.Parametric_Circuit(state,
                                                   jnp.array(self.hamiltonian),
                                                   self.energy_shift, self.weights,
                                                   self.noise_values, Nshots=self.n_shots)

        self.prev_energy = self.get_energy(x0)[0]


        if self.state_with_angles:
            return state.reshape(-1).to(self.device)

        else:
            state = state[:, :self.num_qubits + 3]
            return state.reshape(-1).to(self.device)

    def initial_ep(self):
        print("This is before any episodes. We're not training yet.", flush=True)
        #
        #
        #
        #
        #

    def make_circuit(self, thetas=None):
        """
        based on the angle of first rotation gate we decide if any rotation at
        a given qubit is present i.e.
        if thetas[0, i] == 0 then there is no rotation gate on the Control quibt
        if thetas[1, i] == 0 then there is no rotation gate on the NOT quibt
        CNOT gate have priority over rotations when both will be present in the given slot
        """
        state = self.state.clone()
        if thetas is None:
            thetas = state[:, self.num_qubits + 3:]

        circuit = ParametricQuantumCircuit(self.num_qubits)

        for i in range(self.num_layers):

            cnot_pos = np.where(state[i][0:self.num_qubits] == 1)
            targ = cnot_pos[0]
            ctrl = cnot_pos[1]

            if len(ctrl) != 0:
                for r in range(len(ctrl)):
                    circuit.add_gate(CNOT(ctrl[r], targ[r]))
            rot_pos = np.where(state[i][self.num_qubits: self.num_qubits + 3] == 1)

            rot_direction_list, rot_qubit_list = rot_pos[0], rot_pos[1]

            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    rot_qubit = rot_qubit_list[pos]
                    if r == 0:
                        circuit.add_parametric_RX_gate(rot_qubit,
                                                       thetas[i][0][rot_qubit])  
                    elif r == 1:
                        circuit.add_parametric_RY_gate(rot_qubit, thetas[i][1][rot_qubit])
                    elif r == 2:
                        circuit.add_parametric_RZ_gate(rot_qubit, thetas[i][2][rot_qubit])
                    else:
                        print(f'rot-axis = {r} is in invalid')
                        assert r > 2
        return circuit

    def R_gate(self, qubit, axis, angle):
        if axis == 'X' or axis == 'x' or axis == 1:
            return RX(qubit, angle)
        elif axis == 'Y' or axis == 'y' or axis == 2:
            return RY(qubit, angle)
        elif axis == 'Z' or axis == 'z' or axis == 3:
            return RZ(qubit, angle)
        else:
            print("Wrong gate")
            return 1












    def get_energy(self, angles=None):

        energy = vc_tc.get_exp_val(angles, self.param_circ, self.param_circ.sigma).__array__()


        energy_noiseless = vc_tc.get_noiseless_exp_val(angles, self.param_circ).__array__()




        return energy, energy_noiseless


    def adam_spsa_v2(self, angles):


        x0 = jnp.array(angles, dtype=jnp.float32)


        ener_fn = partial(vc_tc.get_exp_val, param_circ=self.param_circ)


        #


        res = vc_h.min_spsa_v2(ener_fn, x0, maxfev=self.maxfev, **self.options)




        #
        #
        #






        return res['x'] 

    def cobyla_min(self, angles):

        x0 = jnp.array(angles, dtype=jnp.float32)


        ener_fn = partial(vc_tc.get_exp_val, param_circ=self.param_circ)

        #
        #

        #
        #

        result_cobyla = minimize(fun=ener_fn, x0=x0, method='COBYLA', options={'maxiter': 1000})

        return result_cobyla['x']

    def adam_spsa_3(self, angles):

        x0 = jnp.array(angles, dtype=jnp.float32)


        cost3 = partial(vc_tc.get_exp_val, param_circ=self.param_circ,
                        sigma=self.param_circ.sigma)

        cost2 = partial(vc_tc.get_exp_val, param_circ=self.param_circ,
                        sigma=self.param_circ.sigma * np.sqrt(10))

        cost1 = partial(vc_tc.get_exp_val, param_circ=self.param_circ,
                        sigma=self.param_circ.sigma * 10)

        result_spsa = vc_h.min_spsa3_v2(fun1=cost1, fun2=cost2, fun3=cost3,
                                        x0=x0, **self.maxfevs,
                                        **self.options)

        return result_spsa["x"]

    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #

    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #

    def reward_fn(self, energy):
        if self.fn_type == "staircase":
            return (0.2 * (self.error < 15 * self.done_threshold) +
                    0.4 * (self.error < 10 * self.done_threshold) +
                    0.6 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold)) / 2.2
        elif self.fn_type == "two_step":
            return (0.001 * (self.error < 5 * self.done_threshold) +
                    1.0 * (self.error < self.done_threshold)) / 1.001
        elif self.fn_type == "two_step_end":
            max_depth = self.step_counter == (self.num_layers - 1)
            if ((self.error < self.done_threshold) or max_depth):
                return (0.001 * (self.error < 5 * self.done_threshold) +
                        1.0 * (self.error < self.done_threshold)) / 1.001
            else:
                return 0.0
        elif self.fn_type == "naive":
            return 0. + 1. * (self.error < self.done_threshold)
        elif self.fn_type == "incremental":
            return (self.prev_energy - energy) / abs(self.prev_energy - self.min_eig)
        elif self.fn_type == "incremental_clipped":
            return np.clip((self.prev_energy - energy) / abs(self.prev_energy - self.min_eig), -1, 1)
        elif self.fn_type == "nive_fives":
            max_depth = self.step_counter == (self.num_layers - 1)
            if (self.error < self.done_threshold):
                rwd = 5.
            elif max_depth:
                rwd = -5.
            else:
                rwd = 0.
            return rwd

        elif self.fn_type == "incremental_with_fixed_ends":

            max_depth = self.step_counter == (self.num_layers - 1)
            if (self.error < self.done_threshold):
                rwd = 5.
            elif max_depth:
                rwd = -5.
            else:
                rwd = np.clip((self.prev_energy - energy) / abs(self.prev_energy - self.min_eig), -1, 1)
            return rwd

        elif self.fn_type == "log":
            return -np.log(1 - (energy / self.min_eig))

        elif self.fn_type == "log_to_ground":
            return -np.log(abs(energy - self.min_eig))

        elif self.fn_type == "log_to_threshold":
            if self.error < self.done_threshold + 1e-5:
                rwd = 11
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd

        elif self.fn_type == "log_to_threshold_bigger_1000_end":
            if self.error < self.done_threshold + 1e-5:
                rwd = 1000
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd

        elif self.fn_type == "log_to_threshold_bigger_end_no_repeat_actions":
            if self.current_action == self.previous_action:
                return -1  
            elif self.error < self.done_threshold + 1e-5:
                rwd = 20
            else:
                rwd = -np.log(abs(self.error - self.done_threshold))
            return rwd

        elif self.fn_type == "log_neg_punish":
            return -np.log(1 - (energy / self.min_eig)) - 5

        elif self.fn_type == "end_energy":
            max_depth = self.step_counter == (self.num_layers - 1)

            if ((self.error < self.done_threshold) or max_depth):
                rwd = (self.max_eig - energy) / (abs(self.min_eig) + abs(self.max_eig))
            else:
                rwd = 0.0

        elif self.fn_type == "hybrid_reward":
            path = 'threshold_crossed.npy'
            if os.path.exists(path):

                threshold_pass_info = np.load(path)
                if threshold_pass_info > 8:
                    max_depth = self.step_counter == (self.num_layers - 1)
                    if (self.error < self.done_threshold):
                        rwd = 5.
                    elif max_depth:
                        rwd = -5.
                    else:
                        rwd = np.clip((self.prev_energy - energy) / abs(self.prev_energy - self.min_eig), -1, 1)
                    return rwd
                else:
                    if self.error < self.done_threshold + 1e-5:
                        rwd = 11
                    else:
                        rwd = -np.log(abs(self.error - self.done_threshold))
                    return rwd
            else:
                np.save('threshold_crossed.npy', 0)

        elif self.fn_type == "cnot_reduce":
            max_depth = self.step_counter == (self.num_layers - 1)
            if (self.error < self.done_threshold):
                rwd = self.num_layers - self.cnot_rwd_weight * self.current_number_of_cnots
            elif max_depth:
                rwd = -5.
            else:
                rwd = np.clip((self.prev_energy - energy) / abs(self.prev_energy - self.min_eig), -1, 1)
            return

    def illegal_action_new(self):
        action = self.current_action
        illegal_action = self.illegal_actions
        ctrl, targ = action[0], (action[0] + action[1]) % self.num_qubits
        rot_qubit, rot_axis = action[2], action[3]

        if ctrl < self.num_qubits:
            are_you_empty = sum([sum(l) for l in illegal_action])

            if are_you_empty != 0:
                for ill_ac_no, ill_ac in enumerate(illegal_action):

                    if len(ill_ac) != 0:
                        ill_ac_targ = (ill_ac[0] + ill_ac[1]) % self.num_qubits

                        if ill_ac[2] == self.num_qubits:

                            if ctrl == ill_ac[0] or ctrl == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break

                            elif targ == ill_ac[0] or targ == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break

                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                        else:
                            if ctrl == ill_ac[2]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break

                            elif targ == ill_ac[2]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
            else:
                illegal_action[0] = action

        if rot_qubit < self.num_qubits:
            are_you_empty = sum([sum(l) for l in illegal_action])

            if are_you_empty != 0:
                for ill_ac_no, ill_ac in enumerate(illegal_action):

                    if len(ill_ac) != 0:
                        ill_ac_targ = (ill_ac[0] + ill_ac[1]) % self.num_qubits

                        if ill_ac[0] == self.num_qubits:

                            if rot_qubit == ill_ac[2] and rot_axis != ill_ac[3]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action

                                        break

                            elif rot_qubit != ill_ac[2]:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                        else:
                            if rot_qubit == ill_ac[0]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break

                            elif rot_qubit == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break

                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
            else:
                illegal_action[0] = action

        for indx in range(self.num_qubits):
            for jndx in range(indx + 1, self.num_qubits):
                if illegal_action[indx] == illegal_action[jndx]:
                    if jndx != indx + 1:
                        illegal_action[indx] = []
                    else:
                        illegal_action[jndx] = []
                    break

        for indx in range(self.num_qubits - 1):
            if len(illegal_action[indx]) == 0:
                illegal_action[indx] = illegal_action[indx + 1]
                illegal_action[indx + 1] = []

        illegal_action_decode = []
        for key, contain in dictionary_of_actions(self.num_qubits).items():
            for ill_action in illegal_action:
                if ill_action == contain:
                    illegal_action_decode.append(key)
        self.illegal_actions = illegal_action
        return illegal_action_decode


if __name__ == "__main__":
    pass