import jax.numpy as jnp
import numpy as np
from environment.VQE_files.quant.quant_lib_jax import I, CX, X, Y, Z

from jax import jit

dtype = jnp.complex64

# problem = "10max"

encoding = '0xy1'

#6 rotations, 2 cnots
ops_2q = np.zeros((9,2,4,4), dtype = dtype)

ops_2q[0,0,:,:] = I(0,2)  #Dep_0_4_sing_mumbai_ptm

ops_2q[0,1,:,:] = X(0,2)  #X_0_4_sing_mumbai_ptm

ops_2q[1,0,:,:] = I(0,2)  #Dep_0_4_sing_mumbai_ptm

ops_2q[1,1,:,:] = Y(0,2)  #Y_0_4_sing_mumbai_ptm

ops_2q[2,0,:,:] = I(0,2)  #Dep_0_4_sing_mumbai_ptm

ops_2q[2,1,:,:] = Z(0,2)  #Z_0_4_sing_mumbai_ptm

##

ops_2q[3,0,:,:] = I(0,2)  #Dep_1_4_sing_mumbai_ptm

ops_2q[3,1,:,:] = X(1,2)  #X_1_4_sing_mumbai_ptm

ops_2q[4,0,:,:] = I(0,2)  #Dep_1_4_sing_mumbai_ptm

ops_2q[4,1,:,:] = Y(1,2)  #Y_1_4_sing_mumbai_ptm

ops_2q[5,0,:,:] = I(0,2)  #Dep_1_4_sing_mumbai_ptm

ops_2q[5,1,:,:] = Z(1,2)  #Z_1_4_sing_mumbai_ptm

##

ops_2q[6,0,:,:] = CX(0,1,2) #cx_01_4_two_mumbai_ptm

ops_2q[7,0,:,:] = CX(1,0,2) #cx_02_4_two_mumbai_ptm

ops_2q[8,0,:,:] = I(0,2)

ops_2q = jnp.array(ops_2q)


H_2_ptm = jnp.array(  np.load(f"./PTM_files/q2/{encoding}/H_H2_0p7414_jw_2q_ptm.npy"))

rho0_2_ptm = jnp.array(np.load(f"./PTM_files/q2/{encoding}/rho0_2_ptm.npy"))

# if problem == "10max":
print("running 10max 2q problem")
ReadOut_ptm_2 = np.load(
    f"./PTM_files/mumbai/q2/max/{encoding}/ReadOut_ptm_mumbai_2q_max.npy")

I_ptm_0_2 =  np.load(
    f"./PTM_files/mumbai/q2/10max/{encoding}/I_ptm_0_2_mumbai_10max.npy")

I_ptm_1_2 =  np.load(
    f"./PTM_files/mumbai/q2/10max/{encoding}/I_ptm_1_2_mumbai_10max.npy")

X_ptm_0_2 =  np.load(
    f"./PTM_files/mumbai/q2/10max/{encoding}/X_ptm_0_2_mumbai_10max.npy")

X_commute_ptm_0_2 =  np.load(
    f"./PTM_files/mumbai/q2/10max/{encoding}/X_commute_ptm_0_2_mumbai_10max.npy")

# print(X_ptm_0_2)

Y_ptm_0_2 =  np.load(
    f"./PTM_files/mumbai/q2/10max/{encoding}/Y_ptm_0_2_mumbai_10max.npy")

Y_commute_ptm_0_2 =  np.load(
    f"./PTM_files/mumbai/q2/10max/{encoding}/Y_commute_ptm_0_2_mumbai_10max.npy")

Z_ptm_0_2 =  np.load(
    f"./PTM_files/mumbai/q2/10max/{encoding}/Z_ptm_0_2_mumbai_10max.npy")

Z_commute_ptm_0_2 =  np.load(
    f"./PTM_files/mumbai/q2/10max/{encoding}/Z_commute_ptm_0_2_mumbai_10max.npy")

X_ptm_1_2 =  np.load(
    f"./PTM_files/mumbai/q2/10max/{encoding}/X_ptm_1_2_mumbai_10max.npy")

X_commute_ptm_1_2 =  np.load(
    f"./PTM_files/mumbai/q2/10max/{encoding}/X_commute_ptm_1_2_mumbai_10max.npy")

Y_ptm_1_2 =  np.load(
    f"./PTM_files/mumbai/q2/10max/{encoding}/Y_ptm_1_2_mumbai_10max.npy")

Y_commute_ptm_1_2 =  np.load(
    f"./PTM_files/mumbai/q2/10max/{encoding}/Y_commute_ptm_1_2_mumbai_10max.npy")

Z_ptm_1_2 =  np.load(
    f"./PTM_files/mumbai/q2/10max/{encoding}/Z_ptm_1_2_mumbai_10max.npy")

Z_commute_ptm_1_2 =  np.load(
    f"./PTM_files/mumbai/q2/10max/{encoding}/Z_commute_ptm_1_2_mumbai_10max.npy")

cx_ptm_01_2 = np.load(
    f"./PTM_files/mumbai/q2/10max/{encoding}/cx_ptm_01_2_two_mumbai_10max.npy")

cx_ptm_10_2 = np.load(
    f"./PTM_files/mumbai/q2/10max/{encoding}/cx_ptm_10_2_two_mumbai_10max.npy")


ops_ptm_2q = np.zeros((9,3,16,16), dtype = dtype)

ops_ptm_2q[0,0,:,:] = I_ptm_0_2

ops_ptm_2q[0,1,:,:] = X_ptm_0_2

ops_ptm_2q[0,2,:,:] = X_commute_ptm_0_2

ops_ptm_2q[1,0,:,:] = I_ptm_0_2

ops_ptm_2q[1,1,:,:] = Y_ptm_0_2

ops_ptm_2q[1,2,:,:] = Y_commute_ptm_0_2

ops_ptm_2q[2,0,:,:] = I_ptm_0_2

ops_ptm_2q[2,1,:,:] = Z_ptm_0_2

ops_ptm_2q[2,2,:,:] = Z_commute_ptm_0_2

##

ops_ptm_2q[3,0,:,:] = I_ptm_1_2

ops_ptm_2q[3,1,:,:] = X_ptm_1_2

ops_ptm_2q[3,2,:,:] = X_commute_ptm_1_2

ops_ptm_2q[4,0,:,:] = I_ptm_1_2

ops_ptm_2q[4,1,:,:] = Y_ptm_1_2

ops_ptm_2q[4,2,:,:] = Y_commute_ptm_1_2

ops_ptm_2q[5,0,:,:] = I_ptm_1_2

ops_ptm_2q[5,1,:,:] = Z_ptm_1_2

ops_ptm_2q[5,2,:,:] = Z_commute_ptm_1_2
##

ops_ptm_2q[6,0,:,:] = cx_ptm_01_2

ops_ptm_2q[7,0,:,:] = cx_ptm_10_2


ops_ptm_2q[8,0,:,:] = ReadOut_ptm_2

ops_ptm_2q = jnp.array(ops_ptm_2q)



def Rotate_PTM(angle, I_ptm, pauli_ptm,commute_ptm):
    theta = angle/2
    cos = jnp.cos(theta)
    sin = jnp.sin(theta)
    cossq = cos**2
    sinsq = sin**2
    cossin = cos*sin
    return cossq*I_ptm +1j*cossin*commute_ptm +sinsq*pauli_ptm

@jit
def get_mop_ptm_2q(mId, mPauli, mPauliCommute , angle):
    #Kraus representation of a rotation gate, Lambda(rho) is
    # cossq(theta)*rho + 1j*cos(theta)*sin(theta)*[rho,mPauli] + sinsq(theta)*mPauli*rho*mPauli
    #so when converted to PTM it retains this form, but each term involves a pre-computed matrix
    theta = angle/2
    cos = jnp.cos(theta)
    sin = jnp.sin(theta)
    cossq = cos ** 2
    sinsq = sin ** 2
    cossin = cos * sin
    return (cossq * mId) + (1j* cossin * mPauliCommute) + (sinsq * mPauli)

@jit
def get_m_ptm_2q(op_num, angle):
    return get_mop_ptm_2q( ops_ptm_2q[op_num,0,:,:], ops_ptm_2q[op_num,1,:,:],ops_ptm_2q[op_num,2,:,:], angle )

@jit
def mv_16(a,b):
    return a@b

@jit
def get_mop_2q(m1, m2, angle):
    theta = angle/2
    return jnp.cos(theta)*m1 - 1j*jnp.sin(theta)*m2

@jit
def get_m_2q(op_num, angle):
    return get_mop_2q( ops_2q[op_num,0,:,:], ops_2q[op_num,1,:,:], angle )


@jit
def mv_4(a,b):
    return a@b
