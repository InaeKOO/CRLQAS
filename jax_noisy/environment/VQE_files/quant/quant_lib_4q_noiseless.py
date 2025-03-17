
import os, sys
import jax.numpy as jnp
import numpy as np
from environment.VQE_files.quant.quant_lib_jax import I, CX, X, Y, Z
from jax import jit

encoding = '0xy1'

ops_4q = np.zeros((25,2,16,16), dtype = np.complex64)

ops_4q[0,0,:,:] = I(0,4)  #Dep_0_4_ptm

ops_4q[0,1,:,:] = X(0,4)  #X_0_4_ptm

ops_4q[1,0,:,:] = I(0,4)  #Dep_0_4_ptm

ops_4q[1,1,:,:] = Y(0,4)  #Y_0_4_ptm

ops_4q[2,0,:,:] = I(0,4)  #Dep_0_4_ptm

ops_4q[2,1,:,:] = Z(0,4)  #Z_0_4_ptm

##

ops_4q[3,0,:,:] = I(0,4)  #Dep_1_4_ptm

ops_4q[3,1,:,:] = X(1,4)  #X_1_4_ptm

ops_4q[4,0,:,:] = I(0,4)  #Dep_1_4_ptm

ops_4q[4,1,:,:] = Y(1,4)  #Y_1_4_ptm

ops_4q[5,0,:,:] = I(0,4)  #Dep_1_4_ptm

ops_4q[5,1,:,:] = Z(1,4)  #Z_1_4_ptm

##

ops_4q[6,0,:,:] = I(0,4)  #Dep_2_4_ptm

ops_4q[6,1,:,:] = X(2,4)  #X_2_4_ptm

ops_4q[7,0,:,:] = I(0,4)  #Dep_2_4_ptm

ops_4q[7,1,:,:] = Y(2,4)  #Y_2_4_ptm

ops_4q[8,0,:,:] = I(0,4)  #Dep_2_4_ptm

ops_4q[8,1,:,:] = Z(2,4)  #Z_2_4_ptm

##

ops_4q[9,0,:,:]  = I(0,4)  #Dep_3_4_ptm

ops_4q[9,1,:,:]  = X(3,4)  #X_3_4_ptm

ops_4q[10,0,:,:] = I(0,4)  #Dep_3_4_ptm

ops_4q[10,1,:,:] = Y(3,4)  #Y_3_4_ptm

ops_4q[11,0,:,:] = I(0,4)  #Dep_3_4_ptm

ops_4q[11,1,:,:] = Z(3,4)  #Z_3_4_ptm

##

ops_4q[12,0,:,:] = CX(0,1,4) #cx_01_4_ptm

ops_4q[13,0,:,:] = CX(0,2,4) #cx_02_4_ptm

ops_4q[14,0,:,:] = CX(0,3,4) #cx_03_4_ptm

#

ops_4q[15,0,:,:] = CX(1,0,4) #cx_10_4_ptm

ops_4q[16,0,:,:] = CX(1,2,4) #cx_12_4_ptm

ops_4q[17,0,:,:] = CX(1,3,4) #cx_13_4_ptm

#

ops_4q[18,0,:,:] = CX(2,0,4) #cx_20_4_ptm

ops_4q[19,0,:,:] = CX(2,1,4) #cx_21_4_ptm

ops_4q[20,0,:,:] = CX(2,3,4) #cx_23_4_ptm

#

ops_4q[21,0,:,:] = CX(3,0,4) #cx_30_4_ptm

ops_4q[22,0,:,:] = CX(3,1,4) #cx_31_4_ptm

ops_4q[23,0,:,:] = CX(3,2,4) #cx_32_4_ptm

ops_4q[24,0,:,:] = I(0,4)

ops_4q = jnp.array(ops_4q)


cwd = os.getcwd()
print(cwd)

H_4_ptm = jnp.array( np.load(f"./PTM_files/q4/{encoding}/H_LiH_3p4_parity_4q_ptm.npy"))

rho0_4_ptm = jnp.array( np.load(f"./PTM_files/q4/{encoding}/rho0_4_ptm.npy"))

ReadOut_ptm_4 = I(0,8)

I_ptm_0_4 =  I(0,8)

I_ptm_1_4=  I(0,8)

I_ptm_2_4 =  I(0,8)

I_ptm_3_4=  I(0,8)

cx_ptm_01_4 =  np.load(f"./PTM_files/q4/{encoding}/cx_ptm_01_4.npy")

cx_ptm_02_4 =  np.load(f"./PTM_files/q4/{encoding}/cx_ptm_02_4.npy")

cx_ptm_03_4=  np.load(f"./PTM_files/q4/{encoding}/cx_ptm_03_4.npy")

cx_ptm_10_4 =  np.load(f"./PTM_files/q4/{encoding}/cx_ptm_10_4.npy")

cx_ptm_12_4 =  np.load(f"./PTM_files/q4/{encoding}/cx_ptm_12_4.npy")

cx_ptm_13_4 =  np.load(f"./PTM_files/q4/{encoding}/cx_ptm_13_4.npy")

cx_ptm_20_4 =  np.load(f"./PTM_files/q4/{encoding}/cx_ptm_20_4.npy")

cx_ptm_21_4 =  np.load(f"./PTM_files/q4/{encoding}/cx_ptm_21_4.npy")

cx_ptm_23_4 =  np.load(f"./PTM_files/q4/{encoding}/cx_ptm_23_4.npy")

cx_ptm_30_4 =  np.load(f"./PTM_files/q4/{encoding}/cx_ptm_30_4.npy")

cx_ptm_31_4 =  np.load(f"./PTM_files/q4/{encoding}/cx_ptm_31_4.npy")

cx_ptm_32_4 =  np.load(f"./PTM_files/q4/{encoding}/cx_ptm_32_4.npy")

X_ptm_0_4 =  np.load(f"./PTM_files/q4/{encoding}/X_ptm_0_4.npy")

Y_ptm_0_4 =  np.load(f"./PTM_files/q4/{encoding}/Y_ptm_0_4.npy")

Z_ptm_0_4 =  np.load(f"./PTM_files/q4/{encoding}/Z_ptm_0_4.npy")

X_ptm_1_4 =  np.load(f"./PTM_files/q4/{encoding}/X_ptm_1_4.npy")

Y_ptm_1_4 =  np.load(f"./PTM_files/q4/{encoding}/Y_ptm_1_4.npy")

Z_ptm_1_4 =  np.load(f"./PTM_files/q4/{encoding}/Z_ptm_1_4.npy")

X_ptm_2_4  =  np.load(f"./PTM_files/q4/{encoding}/X_ptm_2_4.npy")

Y_ptm_2_4 =  np.load(f"./PTM_files/q4/{encoding}/Y_ptm_2_4.npy")

Z_ptm_2_4 =  np.load(f"./PTM_files/q4/{encoding}/Z_ptm_2_4.npy")

X_ptm_3_4 =  np.load(f"./PTM_files/q4/{encoding}/X_ptm_3_4.npy")

Y_ptm_3_4 =  np.load(f"./PTM_files/q4/{encoding}/Y_ptm_3_4.npy")

Z_ptm_3_4 =  np.load(f"./PTM_files/q4/{encoding}/Z_ptm_3_4.npy")

X_commute_ptm_0_4 =  np.load(f"./PTM_files/q4/{encoding}/X_commute_ptm_0_4.npy")

Y_commute_ptm_0_4 =  np.load(f"./PTM_files/q4/{encoding}/Y_commute_ptm_0_4.npy")

Z_commute_ptm_0_4 =  np.load(f"./PTM_files/q4/{encoding}/Z_commute_ptm_0_4.npy")

X_commute_ptm_1_4 =  np.load(f"./PTM_files/q4/{encoding}/X_commute_ptm_1_4.npy")

Y_commute_ptm_1_4 =  np.load(f"./PTM_files/q4/{encoding}/Y_commute_ptm_1_4.npy")

Z_commute_ptm_1_4 =  np.load(f"./PTM_files/q4/{encoding}/Z_commute_ptm_1_4.npy")

X_commute_ptm_2_4  =  np.load(f"./PTM_files/q4/{encoding}/X_commute_ptm_2_4.npy")

Y_commute_ptm_2_4 =  np.load(f"./PTM_files/q4/{encoding}/Y_commute_ptm_2_4.npy")

Z_commute_ptm_2_4 =  np.load(f"./PTM_files/q4/{encoding}/Z_commute_ptm_2_4.npy")

X_commute_ptm_3_4 =  np.load(f"./PTM_files/q4/{encoding}/X_commute_ptm_3_4.npy")

Y_commute_ptm_3_4 =  np.load(f"./PTM_files/q4/{encoding}/Y_commute_ptm_3_4.npy")

Z_commute_ptm_3_4 =  np.load(f"./PTM_files/q4/{encoding}/Z_commute_ptm_3_4.npy")



ops_ptm_4q = np.zeros((25,3,256,256), dtype = np.complex64)

ops_ptm_4q[0,0,:,:] = I_ptm_0_4

ops_ptm_4q[0,1,:,:] = X_ptm_0_4

ops_ptm_4q[0,2,:,:] = X_commute_ptm_0_4


ops_ptm_4q[1,0,:,:] = I_ptm_0_4

ops_ptm_4q[1,1,:,:] = Y_ptm_0_4

ops_ptm_4q[1,2,:,:] = Y_commute_ptm_0_4

ops_ptm_4q[2,0,:,:] = I_ptm_0_4

ops_ptm_4q[2,1,:,:] = Z_ptm_0_4

ops_ptm_4q[2,2,:,:] = Z_commute_ptm_0_4


##

ops_ptm_4q[3,0,:,:] = I_ptm_1_4

ops_ptm_4q[3,1,:,:] = X_ptm_1_4

ops_ptm_4q[3,2,:,:] = X_commute_ptm_1_4

ops_ptm_4q[4,0,:,:] = I_ptm_1_4

ops_ptm_4q[4,1,:,:] = Y_ptm_1_4

ops_ptm_4q[4,2,:,:] = Y_commute_ptm_1_4

ops_ptm_4q[5,0,:,:] = I_ptm_1_4

ops_ptm_4q[5,1,:,:] = Z_ptm_1_4

ops_ptm_4q[5,2,:,:] = Z_commute_ptm_1_4


##

ops_ptm_4q[6,0,:,:] = I_ptm_2_4

ops_ptm_4q[6,1,:,:] = X_ptm_2_4

ops_ptm_4q[6,2,:,:] = X_commute_ptm_2_4


ops_ptm_4q[7,0,:,:] = I_ptm_2_4

ops_ptm_4q[7,1,:,:] = Y_ptm_2_4

ops_ptm_4q[7,2,:,:] = Y_commute_ptm_2_4


ops_ptm_4q[8,0,:,:] = I_ptm_2_4

ops_ptm_4q[8,1,:,:] = Z_ptm_2_4

ops_ptm_4q[8,2,:,:] = Z_commute_ptm_2_4

##

ops_ptm_4q[9,0,:,:] = I_ptm_3_4

ops_ptm_4q[9,1,:,:] = X_ptm_3_4

ops_ptm_4q[9,2,:,:] = X_commute_ptm_3_4


ops_ptm_4q[10,0,:,:] = I_ptm_3_4

ops_ptm_4q[10,1,:,:] = Y_ptm_3_4

ops_ptm_4q[10,2,:,:] = Y_commute_ptm_3_4


ops_ptm_4q[11,0,:,:] = I_ptm_3_4

ops_ptm_4q[11,1,:,:] = Z_ptm_3_4

ops_ptm_4q[11,2,:,:] = Z_commute_ptm_3_4


##

ops_ptm_4q[12,0,:,:] = cx_ptm_01_4

ops_ptm_4q[13,0,:,:] = cx_ptm_02_4

ops_ptm_4q[14,0,:,:] = cx_ptm_03_4

#

ops_ptm_4q[15,0,:,:] = cx_ptm_10_4

ops_ptm_4q[16,0,:,:] = cx_ptm_12_4

ops_ptm_4q[17,0,:,:] = cx_ptm_13_4

#

ops_ptm_4q[18,0,:,:] = cx_ptm_20_4

ops_ptm_4q[19,0,:,:] = cx_ptm_21_4

ops_ptm_4q[20,0,:,:] = cx_ptm_23_4

#

ops_ptm_4q[21,0,:,:] = cx_ptm_30_4

ops_ptm_4q[22,0,:,:] = cx_ptm_31_4

ops_ptm_4q[23,0,:,:] = cx_ptm_32_4

ops_ptm_4q[24,0,:,:] = ReadOut_ptm_4

ops_ptm_4q = jnp.array(ops_ptm_4q)

@jit
def get_mop_ptm_4q(mId, mPauli, mPauliCommute , angle):
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
def get_m_ptm_4q(op_num, angle):
    return get_mop_ptm_4q( ops_ptm_4q[op_num,0,:,:], ops_ptm_4q[op_num,1,:,:],ops_ptm_4q[op_num,2,:,:], angle )


@jit
def mv_256(a,b):
    return a@b

@jit
def get_mop_4q(m1, m2, angle):
    theta = angle/2
    return jnp.cos(theta)*m1 - 1j*jnp.sin(theta)*m2

@jit
def get_m_4q(op_num, angle):
    return get_mop_4q( ops_4q[op_num,0,:,:], ops_4q[op_num,1,:,:], angle )


@jit
def mv_16(a,b):
    return a@b

