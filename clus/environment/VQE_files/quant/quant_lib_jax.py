import jax

from jax import jit, vmap

import jax.numpy as jnp

import math

import numpy as np


dtype = jnp.complex64

bs0 = jnp.array([[1],[0]],dtype = dtype)

bs1 = jnp.array([[0],[1]],dtype = dtype)


P0 = bs0 @ bs0.conj().T

P1 = bs1 @ bs1.conj().T


sx = jnp.array([[0,1],[1,0]], dtype = dtype)
sy = jnp.array([[0,-1j],[1j,0]], dtype = dtype)
sz = jnp.array([[1,0],[0,-1]], dtype = dtype)

Id = jnp.array([[1,0],[0,1]], dtype=dtype)

GrSt = (Id + sz)/2

ExSt = (Id - sz)/2

h = (1/jnp.sqrt(2)) * jnp.array([[1,1],[1,-1]], dtype=dtype)

s = jnp.array([[1,0],[0,1j]], dtype=dtype)

t = jnp.array([[1,0],[0,(1+1j)/jnp.sqrt(2)]], dtype=dtype)




rx = lambda x: jnp.array([[jnp.cos(x/2),-1j*jnp.sin(x/2)],[-1j*jnp.sin(x/2),jnp.cos(x/2)]], dtype=dtype)

ry = lambda x: jnp.array([[jnp.cos(x/2),-jnp.sin(x/2)],[jnp.sin(x/2),jnp.cos(x/2)]], dtype=dtype)

rz = lambda x: jnp.array([[jnp.exp(-1j*x/2),0],[0,jnp.exp(+1j*x/2)]], dtype=dtype)



def X(targ_q, n_qubits = 4):
    U = jnp.array([1], dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = jnp.kron(U, sx)
        else:
            U = jnp.kron(U, Id)
            
    return U

def Y(targ_q, n_qubits = 4):
    U = jnp.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = jnp.kron(U, sy)
        else:
            U = jnp.kron(U, Id)
            
    return U

def Z(targ_q, n_qubits = 4):
    U = jnp.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = jnp.kron(U, sz)
        else:
            U = jnp.kron(U, Id)
            
    return U

def I(targ_q, n_qubits = 4):
    U = jnp.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = jnp.kron(U, Id)
        else:
            U = jnp.kron(U, Id)
            
    return U

def H(targ_q, n_qubits = 4):
    U = jnp.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = jnp.kron(U, h)
        else:
            U = jnp.kron(U, Id)
            
    return U

def S(targ_q, n_qubits = 4):
    U = jnp.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = jnp.kron(U, s)
        else:
            U = jnp.kron(U, Id)
            
    return U

def T(targ_q, n_qubits = 4):
    U = jnp.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = jnp.kron(U, t)
        else:
            U = jnp.kron(U, Id)
            
    return U

def Rx(targ_q, rads, n_qubits = 4):
    U = jnp.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = jnp.kron(U, rx(rads))
        else:
            U = jnp.kron(U, Id)
            
    return U

def Ry(targ_q, rads, n_qubits = 4):
    U = jnp.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = jnp.kron(U, ry(rads))
        else:
            U = jnp.kron(U, Id)
            
    return U

def Rz(targ_q, rads, n_qubits = 4):
    U = jnp.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq == targ_q:
            U = jnp.kron(U, rz(rads))
        else:
            U = jnp.kron(U, Id)
            
    return U
        
def CX(k,l,n_qubits = 4):
    
    ctr_op = jnp.array([1],dtype = dtype)
    
    not_op = ctr_op
    
    for iq in range(n_qubits):
        if iq == k:
            ctr_op = jnp.kron(ctr_op, GrSt)
            not_op = jnp.kron(not_op, ExSt)
        elif iq ==l:
            ctr_op = jnp.kron(ctr_op, Id)
            not_op = jnp.kron(not_op, sx)
        else:
            ctr_op = jnp.kron(ctr_op, Id)
            not_op = jnp.kron(not_op, Id)
            

    return ctr_op + not_op


def CZ(k,l,n_qubits = 4):
    h_ =  jnp.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        if iq ==l:
            h_ = jnp.kron(h_,h)
        else:
            h_ = jnp.kron(h_, Id)
    
    return h_ @ CX(k,l, n_qubits) @ h_
    
def unitary_init(n_qubits = 4):
    U = Id
    
    for iq in range(1,n_qubits):
        U = jnp.kron(U, Id)
        
    return U

def unitary_init(n_qubits = 4):
    U = Id
    
    for iq in range(1,n_qubits):
        U = jnp.kron(U, Id)
        
    return U

def state_initializer(n_qubits = 4):
    state = jnp.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        state = jnp.kron(state, bs0)
        
    return state

def ket2dm(state):
    
    return state @ state.conj().T

def dm_initializer(n_qubits = 4):
    
    state = jnp.array([1],dtype = dtype)
    
    for iq in range(n_qubits):
        state = jnp.kron(state, P0)
        
    return state

def apply_unitary(unitary, rho):
    
    return unitary @ rho @ unitary.conj().T

def apply_cptp(K_list, rho):
    
    rho_t = jnp.zeros_like(rho, dtype = dtype)
    
    for K in K_list:
        
        rho_t += K@ rho @ K.conj().T 
        
    return rho_t


def depolarizing(error_prob, targ_qub, n_qubits = 4):
        PId, Px, Py, Pz = 1 - 3*error_prob/4, error_prob/4, error_prob/4, error_prob/4
        
        probabilities = [PId, Px, Py, Pz]
        
        unitaries = [I, X, Y, Z]
        
        K_list = []
        
        for iK in range(len(unitaries)):
            K_list.append(jnp.sqrt(probabilities[iK])*unitaries[iK](targ_qub,n_qubits))
            
        return K_list


def bitflip(error_prob, targ_qub, n_qubits=4):
    PId, Px = 1 - error_prob, error_prob

    probabilities = [PId, Px]

    unitaries = [I, X]

    K_list = []

    for iK in range(len(unitaries)):
        K_list.append(jnp.sqrt(probabilities[iK]) * unitaries[iK](targ_qub, n_qubits))

    return K_list

def thresh_tens(At,trsh = 1e-6):
    jnp.real(At)[jnp.abs(jnp.real(At))<trsh] = 0
    jnp.imag(At)[jnp.abs(jnp.imag(At))<trsh] = 0
    
    return At



def noise_map(noise_value, which_qubits, n_batch):
    n_sqg = len(which_qubits)
    
    prng = np.random.RandomState()
        
    PId = 1 - 3*noise_value/4
    Px, Py, Pz = noise_value/4, noise_value/4, noise_value/4
        
    probabilities = [PId, Px, Py, Pz]
        
    n_mp = np.zeros([n_sqg, n_batch])
        
    for it in range(n_batch):
        for sqg in range(n_sqg):
            n_mp[sqg, it] = prng.choice(4, p=probabilities)
            
    n_mp = np.c_[which_qubits.reshape(n_sqg,1), n_mp]
    # nmp[:,0] = which_qubits
                
    return n_mp


def get_I3(n_qubits, n_batch):
    N = int(2**n_qubits)
    shape = (N,N,n_batch)
    identity_3d = np.zeros(shape, dtype = dtype)
    idx = np.arange(shape[0])
    identity_3d[idx, idx, :] = 1  
    return jax.device_put(identity_3d)

def get_I3_(n_qubits, n_batch):
    # return I(0,n_qubits).repeat(n_batch,1,1)
    return jnp.repeat(I(0,n_qubits)[jnp.newaxis, :, :], n_batch, axis=0)

def u2t(U, n_batch):
    # return U.repeat(n_batch, 1, 1)
    return jnp.repeat(U[jnp.newaxis, :, :], n_batch, axis=0)


def m2t(A, I3):
    T = jnp.einsum('ijk, jl -> ilk', I3, A)
    return T

def nm2t(n_mp,n_qubits):
    N = int(2 ** n_qubits)
    n_sqg = n_mp.shape[0]
    
    n_batch = n_mp.shape[1] - 1
    
    unitaries = [I, X, Y, Z]
    MT = np.zeros([N, N, n_batch, n_sqg], dtype = dtype)
    
    for iB in range(n_batch):
        for iSq in range(n_sqg):
            q_index, u_index = n_mp[iSq, 0], n_mp[iSq, iB + 1]
            q_index, u_index = int(q_index), int(u_index)
            unitary = unitaries[u_index](q_index,n_qubits)
            MT[:, :, iB, iSq] = unitary
            
    return jax.device_put(MT)


def nm2t_(n_mp,n_qubits):
    N = int(2 ** n_qubits)
    n_sqg = n_mp.shape[0]
    
    n_batch = n_mp.shape[1] - 1
    
    unitaries = [I, X, Y, Z]
    MT = np.zeros([n_batch, N, N,  n_sqg], dtype = dtype)
    
    for iB in range(n_batch):
        for iSq in range(n_sqg):
            q_index, u_index = n_mp[iSq, 0], n_mp[iSq, iB + 1]
            q_index, u_index = int(q_index), int(u_index)
            unitary = unitaries[u_index](q_index,n_qubits)
            MT[iB, :, : , iSq] = unitary
            
    return jax.device_put(MT)

def noiseMap2delayed(n_mp,n_qubits):
    N = int(2 ** n_qubits)
    n_sqg = n_mp.shape[0]
    
    n_batch = n_mp.shape[1] - 1
    
    unitaries = [I, X, Y, Z]
    MT = np.zeros([n_batch, N, N], dtype = dtype)
    
    for iB in range(n_batch):
        unitary = I(0, n_qubits)
        for iSq in range(n_sqg):
            q_index, u_index = n_mp[iSq, 0], n_mp[iSq, iB + 1]
            q_index, u_index = int(q_index), int(u_index)
            unitary = unitaries[u_index](q_index,n_qubits) @ unitary
            
        MT[iB, :, : ] = unitary
            
    return jax.device_put(MT)

def get_depol_mask_delayed(noise_value, n_qubits, which_qubits, n_batch = 1000):
    n_mp = noise_map(noise_value, which_qubits, n_batch)
    MT = noiseMap2delayed(n_mp,n_qubits)
    
    return MT



def get_depol_mask(noise_value, n_qubits, which_qubits, n_batch = 1000):
    
    n_mp = noise_map(noise_value, which_qubits, n_batch)
    MT = nm2t(n_mp,n_qubits)
    
    return MT

def get_depol_mask_(noise_value, n_qubits, which_qubits, n_batch = 1000):
    
    n_mp = noise_map(noise_value, which_qubits, n_batch)
    MT = nm2t_(n_mp,n_qubits)
    
    return MT

def tensmm(M0, T):
    M0T = jnp.einsum('ijk, jlk -> ilk ' , M0, T)
    return M0T




# def depolarizing_fn(error_prob):
#         PId, Px, Py, Pz = 1 - 3*error_prob/4, error_prob/4, error_prob/4, error_prob/4
        
#         probabilities = [PId, Px, Py, Pz]
        
#         unitaries = [I, X, Y, Z]
        
#         K_list_fn = []
        
#         for iK in range(len(unitaries)):
#             def K_fun(targ_qub, n_qubits): 
#                 return np.sqrt(probabilities[iK])*unitaries[iK](targ_qub, n_qubits)
            
#             K_list_fn.append(K_fun)
            
#         return K_list_fn


