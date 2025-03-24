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
import scipy.linalg as la 
import scipy.optimize as optimize
import os
import numpy as np
import random
import copy
import curricula
from functools import partial
from scipy.optimize import minimize
from scipy.linalg import sqrtm
from qiskit.quantum_info import Pauli, Clifford, random_clifford, SparsePauliOp, random_density_matrix, DensityMatrix

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" 


try:
    from qulacs import QuantumStateGpu as QuantumState
except ImportError:
    from qulacs import QuantumState

from qulacs import ParametricQuantumCircuit

from typing import List, Callable, Tuple, Optional, Dict
import copy

from scipy.optimize import OptimizeResult
from shadow import *


def random_unitary(d):

    Z = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    Q, R = np.linalg.qr(Z)
    diag_R = np.diag(R)
    phase = diag_R / np.abs(diag_R)
    Q = Q * phase
    I = np.eye(d)
    #for i in range(int(d/2),int(3*d/4)):
    #    I[:, [i+int(d/4),i]] = I[:, [i,i+int(d/4)]]
    return I

def create_phi_plus(n):
    d = 2 **n
    phi_plus = np.zeros((d*d,), dtype=complex)
    for i in range(d):
        phi_plus[i*d + i] = 1.0
    return phi_plus / np.sqrt(d)

def choi_state(U, n):
    d = 2 ** n
    phi_plus = create_phi_plus(n)          # 최대 얽힘 상태 |Φ⁺⟩, shape: (d*d,)
    rho_phi = np.outer(phi_plus, np.conjugate(phi_plus))  # |Φ⁺⟩⟨Φ⁺|, shape: (d^2, d^2)
    
    # I_d ⊗ U 적용: I_d는 d x d 항등행렬, U는 2^n x 2^n 행렬
    IU = np.kron(np.eye(d), U)
    choi = IU @ rho_phi @ IU.conjugate().T
    return choi

def safe_matrix_sqrt(m, tol=1e-8):
    eigenvals, eigenvecs = np.linalg.eigh(m)
    # tol 이하의 음수 고유값은 0으로 처리
    eigenvals = np.where(eigenvals < tol, 0, eigenvals)
    sqrt_eigenvals = np.sqrt(eigenvals)
    return eigenvecs @ np.diag(sqrt_eigenvals) @ eigenvecs.conj().T

class CircuitEnv():

    def __init__(self, conf, device):
        self.random_halt = int(conf['env']['rand_halt'])
        
        
        self.num_qubits = conf['env']['num_qubits']
        print(f"num qubits is {self.num_qubits}")
        self.unitary = random_unitary(2**self.num_qubits)
        print(f"Unitary Matrix is: {self.unitary}")

        self.num_layers = conf['env']['num_layers']
        
        
        self.n_shots =   conf['env']['n_shots'] 
        noise_values = list(filter(None, conf['env']['noise_values']))
        noise_models = ['depolarizing', 'two_depolarizing', 'amplitude_damping']
        self.noise_values = noise_values
        self.noise_models = noise_models[0:len(self.noise_values)]
        if len(self.noise_values) > 0:
            self.phys_noise = True
        else:
            self.phys_noise = False
        self.err_mitig = conf['env']['err_mitig']
        
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

        self.curriculum_dict = curricula.__dict__[conf['env']['curriculum_type']](conf['env'], target_fidelity=1)

        self.curriculum_type = conf['env']['curriculum_type']
        self.device = device
        self.done_threshold = conf['env']['accept_err']
      

        stdout.flush()
        self.state_size = self.num_layers*self.num_qubits*(self.num_qubits+3+3)
        self.step_counter = -1
        self.prev_fidelity = None
        self.moments = [0]*self.num_qubits
        self.illegal_actions = [[]]*self.num_qubits

        self.action_size = 18
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
                self.maxfevs['maxfev1'] = int( conf['non_local_opt']["maxfev1"] )
                self.maxfevs['maxfev2'] = int( conf['non_local_opt']["maxfev2"] )
                self.maxfevs['maxfev3'] = int( conf['non_local_opt']["maxfev3"] )
        else:
            self.global_iters = 0
            self.optim_method = None
            
            



    def step(self, action, train_flag = True) :

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
        print("action: ",action)

        

        if rot_qubit < self.num_qubits:
            gate_tensor = self.moments[ rot_qubit ]
        elif ctrl < self.num_qubits:
            gate_tensor = max( self.moments[ctrl], self.moments[targ] )

        if ctrl < self.num_qubits:
            next_state[gate_tensor][targ][ctrl] = 1
        elif rot_qubit < self.num_qubits:
            next_state[gate_tensor][self.num_qubits+rot_axis-1][rot_qubit] = 1

        if rot_qubit < self.num_qubits:
            self.moments[ rot_qubit ] += 1
        elif ctrl < self.num_qubits:
            max_of_two_moments = max( self.moments[ctrl], self.moments[targ] )
            self.moments[ctrl] = max_of_two_moments +1
            self.moments[targ] = max_of_two_moments +1
            
            
        self.current_action = action
        self.illegal_action_new()

        
        state = self.state.clone()
        thetas = state[:, self.num_qubits+3:]
        rot_pos = (state[:,self.num_qubits: self.num_qubits+3] == 1).nonzero( as_tuple = True )
        angles = thetas[rot_pos]
    
        self.param_circ = ParametricQuantumCircuit(self.num_qubits)
                

        fidelity = self.fidelity()
      
        for i in range(self.num_layers):
            for j in range(3):
                next_state[i][self.num_qubits+3+j,:] = thetas[i][j,:]

        self.state = next_state.clone()
        cnots = self.state[:, :self.num_qubits]
        self.current_number_of_cnots = np.count_nonzero(cnots)

        
        if fidelity > self.curriculum.highest_fidelity and train_flag:
            self.curriculum.highest_fidelity = copy.copy(fidelity)
    
        #self.error = float(abs(self.min_eig-energy))
        self.error = float(self.max_fidelity-fidelity)

        
        #rwd = self.reward_fn(energy)
        rwd = self.reward_fidelity(fidelity)
        self.prev_fidelity = np.copy(fidelity)
        print("error: ",self.error,", reward: ", rwd, ", done_threshold: ", self.done_threshold)


        fidelity_done = int(self.error < self.done_threshold)
        layers_done = self.step_counter == (self.num_layers - 1)
        done = int(fidelity_done or layers_done)
        
        self.previous_action = copy.deepcopy(action)

        
        if self.random_halt:
            if self.step_counter == self.halting_step:
                print(f"Last action of the episode now.")
                print(f"Fidelity is {self.prev_fidelity}", flush=True)
                done = 1
     
        if done:
            self.curriculum.update_threshold(energy_done=fidelity_done)
            self.done_threshold = self.curriculum.get_current_threshold()
            self.curriculum_dict = copy.deepcopy(self.curriculum)
        
        if self.state_with_angles:
            return next_state.view(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done
        else:
            next_state = next_state[:, :self.num_qubits+3]
            return next_state.reshape(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done

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
        state = torch.zeros((self.num_layers, self.num_qubits+3+3, self.num_qubits))
        self.state = state
        
        
        if self.random_halt:
            statistics_generated = np.clip(np.random.negative_binomial(n=40, p=0.6, size=100), 0, 40)
            c = Counter(statistics_generated)
            self.halting_step = c.most_common(1)[0][0]
        
        
        self.current_number_of_cnots = 0
        self.current_action = [self.num_qubits]*4
        self.illegal_actions = [[]]*self.num_qubits
        
        self.step_counter = -1

        self.moments = [0]*self.num_qubits
        self.curriculum = copy.deepcopy(self.curriculum_dict)
        self.done_threshold = copy.deepcopy(self.curriculum.get_current_threshold())

        self.max_fidelity = 1
        
        state = self.state.clone()
        thetas = state[:, self.num_qubits+3:]
        rot_pos = (state[:,self.num_qubits: self.num_qubits+3] == 1).nonzero( as_tuple = True )
        angles = thetas[rot_pos]
        x0 = jnp.array(angles, dtype = jnp.float32)


        self.param_circ = ParametricQuantumCircuit(self.num_qubits)
        self.prev_fidelity = self.fidelity()
        print("Initial fidelity: ", self.prev_fidelity)

        if self.state_with_angles:
            return state.reshape(-1).to(self.device)
            
        else:
            state = state[:, :self.num_qubits+3]
            return state.reshape(-1).to(self.device)

    def initial_ep(self):
        print("This is before any episodes. We're not training yet.",flush=True)

    def make_circuit(self):
        """
        based on the angle of first rotation gate we decide if any rotation at
        a given qubit is present i.e.
        if thetas[0, i] == 0 then there is no rotation gate on the Control quibt
        if thetas[1, i] == 0 then there is no rotation gate on the NOT quibt
        CNOT gate have priority over rotations when both will be present in the given slot
        """
        state = self.state.clone()
        n = self.num_qubits
        qs = QuantumState(n*2)
        circuit = ParametricQuantumCircuit(n*2)

        for i in range(self.num_qubits):
            circuit.add_gate(H(i))
        
        for i in range(self.num_qubits):
            circuit.add_gate(CNOT(i, i+4))
        
        for i in range(self.num_layers):
            
            cnot_pos = np.where(state[i][0:self.num_qubits] == 1)
            targ = cnot_pos[0]
            ctrl = cnot_pos[1]
            
            if len(ctrl) != 0:
                for r in range(len(ctrl)):
                    circuit.add_gate(CNOT(n+ctrl[r], n+targ[r]))
            rot_pos = np.where(state[i][self.num_qubits: self.num_qubits+3] == 1)
            
            rot_direction_list, rot_qubit_list = rot_pos[0], rot_pos[1]
            
            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    rot_qubit = rot_qubit_list[pos]
                    if r == 0:
                        circuit.add_parametric_RX_gate(n+rot_qubit, np.pi/16)
                    elif r == 1:
                        circuit.add_parametric_RY_gate(n+rot_qubit, np.pi/16)
                    elif r == 2:
                        circuit.add_parametric_RZ_gate(n+rot_qubit, np.pi/16)
                    else:
                        print(f'rot-axis = {r} is in invalid')
                        assert r >2
        circuit.update_quantum_state(qs)
        return qs

    def R_gate(self, qubit, axis, angle):
        if axis == 'X' or axis == 'x' or axis == 1:
            return RX(qubit, angle)
        elif axis == 'Y' or axis == 'y' or axis == 2:
            return RY(qubit, angle)
        elif axis == 'Z' or axis == 'z' or axis == 3:
            return RZ(qubit, angle)
        else:
            print("Wrong gate")
            return 13

    def choi_fidelity(self, rho, sigma):
        print("Is density? rho: ", is_density_matrix(rho), " , sigma: ", is_density_matrix(sigma))
        sqrt_rho = safe_matrix_sqrt(rho)
        product = sqrt_rho @ sigma @ sqrt_rho
        sqrt_product = safe_matrix_sqrt(product)
        fid = (np.trace(sqrt_product).real) ** 2  # 미세한 허수부를 제거
        return fid
    
    def classical_shadow(self, num_samples = 100):
        n = self.num_qubits * 2
        dim = 2 ** n
        rho_est_total = np.zeros((dim, dim), dtype=complex)
        for _ in range(num_samples):
            bases = [get_random_basis() for _ in range(n)]
            qs = self.make_circuit()
            meas_circuit = QuantumCircuit(n)
            for qubit in range(n):
                gate = get_measurement_gate(bases[qubit], qubit)
                if gate is not None:
                    meas_circuit.add_gate(gate)
            meas_circuit.update_quantum_state(qs)
            outcome = qs.sampling(1)[0]
            outcome_str = format(outcome, f'0{n}b')
            #print(outcome_str)
            outcomes = [int(bit) for bit in outcome_str]

            # For each qubit, compute its single-qubit estimator.
            estimators = [get_single_qubit_estimator(bases[q], outcomes[q]) for q in range(n)]
            # The snapshot estimator is the tensor product over all qubits.
            rho_snapshot = tensor_list(estimators)
            
            # Accumulate the estimator.
            rho_est_total += rho_snapshot

        return rho_est_total / num_samples
    
    def fidelity(self, angles=None):
        return self.choi_fidelity(choi_state(self.unitary, self.num_qubits), self.classical_shadow())
    

    def reward_fidelity(self, fidelity):
        max_depth = self.step_counter == (self.num_layers - 1)
        if (1-fidelity < self.done_threshold):
            rwd = 5.
        elif max_depth:
            rwd = -5.
        else:
            rwd = np.clip((fidelity-self.prev_fidelity)/abs(self.max_fidelity-self.prev_fidelity),-1,1)
        return rwd

        
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
                        ill_ac_targ = ( ill_ac[0] + ill_ac[1] ) % self.num_qubits
                        
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
                        ill_ac_targ = ( ill_ac[0] + ill_ac[1] ) % self.num_qubits
                        
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
            for jndx in range(indx+1, self.num_qubits):
                if illegal_action[indx] == illegal_action[jndx]:
                    if jndx != indx +1:
                        illegal_action[indx] = []
                    else:
                        illegal_action[jndx] = []
                    break
        
        for indx in range(self.num_qubits-1):
            if len(illegal_action[indx])==0:
                illegal_action[indx] = illegal_action[indx+1]
                illegal_action[indx+1] = []
        
        illegal_action_decode = []
        for key, contain in dictionary_of_actions(self.num_qubits).items():
            for ill_action in illegal_action:
                if ill_action == contain:
                    illegal_action_decode.append(key)
        self.illegal_actions = illegal_action
        return illegal_action_decode




if __name__ == "__main__":
    pass
