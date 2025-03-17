import jax.numpy as jnp
import numpy as np
from environment.VQE_files.quant.quant_lib_jax import I, CX, X, Y, Z

from jax import jit


encoding = '0xy1'



#6 rotations, 2 cnots
ops_3q = np.zeros((16,2,8,8), dtype = np.complex64)

ops_3q[0,0,:,:] = I(0,3)  #Dep_0_4_sing_mumbai_ptm

ops_3q[0,1,:,:] = X(0,3)  #X_0_4_sing_mumbai_ptm

ops_3q[1,0,:,:] = I(0,3)  #Dep_0_4_sing_mumbai_ptm

ops_3q[1,1,:,:] = Y(0,3)  #Y_0_4_sing_mumbai_ptm

ops_3q[2,0,:,:] = I(0,3)  #Dep_0_4_sing_mumbai_ptm

ops_3q[2,1,:,:] = Z(0,3)  #Z_0_4_sing_mumbai_ptm

##

ops_3q[3,0,:,:] = I(0,3)  #Dep_1_4_sing_mumbai_ptm

ops_3q[3,1,:,:] = X(1,3)  #X_1_4_sing_mumbai_ptm

ops_3q[4,0,:,:] = I(0,3)  #Dep_1_4_sing_mumbai_ptm

ops_3q[4,1,:,:] = Y(1,3)  #Y_1_4_sing_mumbai_ptm

ops_3q[5,0,:,:] = I(0,3)  #Dep_1_4_sing_mumbai_ptm

ops_3q[5,1,:,:] = Z(1,3)  #Z_1_4_sing_mumbai_ptm

##

ops_3q[6,0,:,:] = I(0,3)  #Dep_1_4_sing_mumbai_ptm

ops_3q[6,1,:,:] = X(2,3)  #X_1_4_sing_mumbai_ptm

ops_3q[7,0,:,:] = I(0,3)  #Dep_1_4_sing_mumbai_ptm

ops_3q[7,1,:,:] = Y(2,3)  #Y_1_4_sing_mumbai_ptm

ops_3q[8,0,:,:] = I(0,3)  #Dep_1_4_sing_mumbai_ptm

ops_3q[8,1,:,:] = Z(2,3)  #Z_1_4_sing_mumbai_ptm

###


ops_3q[9,0,:,:] = CX(0,1,3) #cx_01_4_two_mumbai_ptm

ops_3q[10,0,:,:] = CX(0,2,3) #cx_02_4_two_mumbai_ptm

####


ops_3q[11,0,:,:] = CX(1,0,3) #cx_01_4_two_mumbai_ptm

ops_3q[12,0,:,:] = CX(1,2,3) #cx_02_4_two_mumbai_ptm

###

ops_3q[13,0,:,:] = CX(2,0,3) #cx_01_4_two_mumbai_ptm

ops_3q[14,0,:,:] = CX(2,1,3) #cx_02_4_two_mumbai_ptm


ops_3q[15,0,:,:] = I(0,3)

ops_3q = jnp.array(ops_3q)


H_3_ptm = jnp.array( np.load(f"./PTM_files/q3/{encoding}/H_H2_0p7414_jw_3q_ptm.npy"))

rho0_3_ptm = jnp.array( np.load(f"./PTM_files/q3/{encoding}/rho0_3_ptm.npy"))

# if problem == "median":

ReadOut_3_ptm = I(0,6)

I_0_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/Dep_ptm_0_3_0p001.npy")

I_1_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/Dep_ptm_1_3_0p001.npy")

I_2_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/Dep_ptm_1_3_0p001.npy")


X_0_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/X_ptm_0_3_0p001.npy")

Y_0_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/Y_ptm_0_3_0p001.npy")

Z_0_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/Z_ptm_0_3_0p001.npy")


X_commute_0_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/X_commute_ptm_0_3_0p001.npy")

Y_commute_0_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/Y_commute_ptm_0_3_0p001.npy")

Z_commute_0_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/Z_commute_ptm_0_3_0p001.npy")


X_1_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/X_ptm_1_3_0p001.npy")

Y_1_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/Y_ptm_1_3_0p001.npy")

Z_1_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/Z_ptm_1_3_0p001.npy")


X_commute_1_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/X_commute_ptm_1_3_0p001.npy")

Y_commute_1_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/Y_commute_ptm_1_3_0p001.npy")

Z_commute_1_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/Z_commute_ptm_1_3_0p001.npy")


X_2_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/X_ptm_2_3_0p001.npy")

Y_2_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/Y_ptm_2_3_0p001.npy")

Z_2_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/Z_ptm_2_3_0p001.npy")


X_commute_2_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/X_commute_ptm_2_3_0p001.npy")

Y_commute_2_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/Y_commute_ptm_2_3_0p001.npy")

Z_commute_2_3_ptm = np.load(f"./PTM_files/q3/{encoding}/0p001/Z_commute_ptm_2_3_0p001.npy")


cx_01_3_ptm = np.load(f"./PTM_files/q3/{encoding}/cx_ptm_01_3.npy")

cx_02_3_ptm = np.load(f"./PTM_files/q3/{encoding}/cx_ptm_02_3.npy")

cx_10_3_ptm = np.load(f"./PTM_files/q3/{encoding}/cx_ptm_10_3.npy")

cx_12_3_ptm = np.load(f"./PTM_files/q3/{encoding}/cx_ptm_12_3.npy")

cx_20_3_ptm = np.load(f"./PTM_files/q3/{encoding}/cx_ptm_20_3.npy")

cx_21_3_ptm = np.load(f"./PTM_files/q3/{encoding}/cx_ptm_21_3.npy")


ops_ptm_3q = np.zeros((16,3,64,64), dtype = np.complex64)

ops_ptm_3q[0,0,:,:] = I_0_3_ptm

ops_ptm_3q[0,1,:,:] = X_0_3_ptm

ops_ptm_3q[0,2,:,:] = X_commute_0_3_ptm

ops_ptm_3q[1,0,:,:] = I_0_3_ptm

ops_ptm_3q[1,1,:,:] = Y_0_3_ptm

ops_ptm_3q[1,2,:,:] = Y_commute_0_3_ptm

ops_ptm_3q[2,0,:,:] = I_0_3_ptm

ops_ptm_3q[2,1,:,:] = Z_0_3_ptm

ops_ptm_3q[2,2,:,:] = Z_commute_0_3_ptm


##

ops_ptm_3q[3,0,:,:] = I_1_3_ptm

ops_ptm_3q[3,1,:,:] = X_1_3_ptm

ops_ptm_3q[3,2,:,:] = X_commute_1_3_ptm


ops_ptm_3q[4,0,:,:] = I_1_3_ptm

ops_ptm_3q[4,1,:,:] = Y_1_3_ptm

ops_ptm_3q[4,2,:,:] = Y_commute_1_3_ptm


ops_ptm_3q[5,0,:,:] = I_1_3_ptm

ops_ptm_3q[5,1,:,:] = Z_1_3_ptm

ops_ptm_3q[5,2,:,:] = Z_commute_1_3_ptm


##

ops_ptm_3q[6,0,:,:] = I_2_3_ptm

ops_ptm_3q[6,1,:,:] = X_2_3_ptm

ops_ptm_3q[6,2,:,:] = X_commute_2_3_ptm


ops_ptm_3q[7,0,:,:] = I_2_3_ptm

ops_ptm_3q[7,1,:,:] = Y_2_3_ptm

ops_ptm_3q[7,2,:,:] = Y_commute_2_3_ptm


ops_ptm_3q[8,0,:,:] = I_2_3_ptm

ops_ptm_3q[8,1,:,:] = Z_2_3_ptm

ops_ptm_3q[8,2,:,:] = Z_commute_2_3_ptm


##

ops_ptm_3q[9,0,:,:] = cx_01_3_ptm

ops_ptm_3q[10,0,:,:] = cx_02_3_ptm

ops_ptm_3q[11,0,:,:] = cx_10_3_ptm

ops_ptm_3q[12,0,:,:] = cx_12_3_ptm

ops_ptm_3q[13,0,:,:] = cx_20_3_ptm

ops_ptm_3q[14,0,:,:] = cx_21_3_ptm

ops_ptm_3q[15,0,:,:] = ReadOut_3_ptm

ops_ptm_3q = jnp.array(ops_ptm_3q)

@jit
def get_mop_ptm_3q(mId, mPauli, mPauliCommute , angle):
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
def get_m_ptm_3q(op_num, angle):
    return get_mop_ptm_3q( ops_ptm_3q[op_num,0,:,:], ops_ptm_3q[op_num,1,:,:],ops_ptm_3q[op_num,2,:,:], angle )

@jit
def mv_64(a,b):
    return a@b

@jit
def get_mop_3q(m1, m2, angle):
    theta = angle/2
    return jnp.cos(theta)*m1 - 1j*jnp.sin(theta)*m2

@jit
def get_m_3q(op_num, angle):
    return get_mop_3q( ops_3q[op_num,0,:,:], ops_3q[op_num,1,:,:], angle )


@jit
def mv_8(a,b):
    return a@b

