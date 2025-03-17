import jax.numpy as jnp
import numpy as np
from environment.VQE_files.quant.quant_lib_jax import I, CX, X, Y, Z

from jax import jit

encoding = '0xy1'

ops_4q = np.zeros((19,2,16,16), dtype = np.complex64)

ops_4q[0,0,:,:] = I(0,4)  #Dep_0_4_sing_ptm

ops_4q[0,1,:,:] = X(0,4)  #X_0_4_sing_ptm

ops_4q[1,0,:,:] = I(0,4)  #Dep_0_4_sing_ptm

ops_4q[1,1,:,:] = Y(0,4)  #Y_0_4_sing_ptm

ops_4q[2,0,:,:] = I(0,4)  #Dep_0_4_sing_ptm

ops_4q[2,1,:,:] = Z(0,4)  #Z_0_4_sing_ptm

##

ops_4q[3,0,:,:] = I(0,4)  #Dep_1_4_sing_ptm

ops_4q[3,1,:,:] = X(1,4)  #X_1_4_sing_ptm

ops_4q[4,0,:,:] = I(0,4)  #Dep_1_4_sing_ptm

ops_4q[4,1,:,:] = Y(1,4)  #Y_1_4_sing_ptm

ops_4q[5,0,:,:] = I(0,4)  #Dep_1_4_sing_ptm

ops_4q[5,1,:,:] = Z(1,4)  #Z_1_4_sing_ptm

##

ops_4q[6,0,:,:] = I(0,4)  #Dep_2_4_sing_ptm

ops_4q[6,1,:,:] = X(2,4)  #X_2_4_sing_ptm

ops_4q[7,0,:,:] = I(0,4)  #Dep_2_4_sing_ptm

ops_4q[7,1,:,:] = Y(2,4)  #Y_2_4_sing_ptm

ops_4q[8,0,:,:] = I(0,4)  #Dep_2_4_sing_ptm

ops_4q[8,1,:,:] = Z(2,4)  #Z_2_4_sing_ptm

##

ops_4q[9,0,:,:]  = I(0,4)  #Dep_3_4_sing_ptm

ops_4q[9,1,:,:]  = X(3,4)  #X_3_4_sing_ptm

ops_4q[10,0,:,:] = I(0,4)  #Dep_3_4_sing_ptm

ops_4q[10,1,:,:] = Y(3,4)  #Y_3_4_sing_ptm

ops_4q[11,0,:,:] = I(0,4)  #Dep_3_4_sing_ptm

ops_4q[11,1,:,:] = Z(3,4)  #Z_3_4_sing_ptm

##

ops_4q[12,0,:,:] = CX(0,1,4) #cx_01_4_two_ptm


#

ops_4q[13,0,:,:] = CX(1,0,4) #cx_10_4_two_ptm

ops_4q[14,0,:,:] = CX(1,2,4) #cx_12_4_two_ptm

ops_4q[15,0,:,:] = CX(1,3,4) #cx_13_4_two_ptm

#


ops_4q[16,0,:,:] = CX(2,1,4) #cx_21_4_two_ptm


#


ops_4q[17,0,:,:] = CX(3,1,4) #cx_31_4_two_ptm

ops_4q[18,0,:,:] = I(0,4)

ops_4q = jnp.array(ops_4q)




H_4_ptm = jnp.array( np.load(f"./PTM_files/ourense/q4/{encoding}/H_H2_0p35_jordan_wigner_4q_ptm.npy"))

rho0_4_ptm = jnp.array( np.load(f"./PTM_files/q4/{encoding}/rho0_4_ptm.npy") )

ReadOut_4_ptm = I(0,8) #np.load(f"./PTM_files/ourense/q4/{encoding}/ReadOut_ptm_ourense_4.npy")

I_ptm_0_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/I_ptm_0_4_ourense.npy")

I_ptm_1_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/I_ptm_1_4_ourense.npy")

I_ptm_2_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/I_ptm_2_4_ourense.npy")

I_ptm_3_4  =  np.load(f"./PTM_files/ourense/q4/{encoding}/I_ptm_3_4_ourense.npy")

cx_ptm_01_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/cx_ptm_01_4_two_ourense.npy")

cx_ptm_10_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/cx_ptm_10_4_two_ourense.npy")

cx_ptm_12_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/cx_ptm_12_4_two_ourense.npy")

cx_ptm_13_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/cx_ptm_13_4_two_ourense.npy")

cx_ptm_21_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/cx_ptm_21_4_two_ourense.npy")

cx_ptm_31_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/cx_ptm_31_4_two_ourense.npy")


X_ptm_0_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/X_ptm_0_4_ourense.npy")

Y_ptm_0_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/Y_ptm_0_4_ourense.npy")

Z_ptm_0_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/Z_ptm_0_4_ourense.npy")

X_ptm_1_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/X_ptm_1_4_ourense.npy")

Y_ptm_1_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/Y_ptm_1_4_ourense.npy")

Z_ptm_1_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/Z_ptm_1_4_ourense.npy")

X_ptm_2_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/X_ptm_2_4_ourense.npy")

Y_ptm_2_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/Y_ptm_2_4_ourense.npy")

Z_ptm_2_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/Z_ptm_2_4_ourense.npy")

X_ptm_3_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/X_ptm_3_4_ourense.npy")

Y_ptm_3_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/Y_ptm_3_4_ourense.npy")

Z_ptm_3_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/Z_ptm_3_4_ourense.npy")



X_commute_ptm_0_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/X_commute_ptm_0_4_ourense.npy")

Y_commute_ptm_0_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/Y_commute_ptm_0_4_ourense.npy")

Z_commute_ptm_0_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/Z_commute_ptm_0_4_ourense.npy")

X_commute_ptm_1_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/X_commute_ptm_1_4_ourense.npy")

Y_commute_ptm_1_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/Y_commute_ptm_1_4_ourense.npy")

Z_commute_ptm_1_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/Z_commute_ptm_1_4_ourense.npy")

X_commute_ptm_2_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/X_commute_ptm_2_4_ourense.npy")

Y_commute_ptm_2_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/Y_commute_ptm_2_4_ourense.npy")

Z_commute_ptm_2_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/Z_commute_ptm_2_4_ourense.npy")

X_commute_ptm_3_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/X_commute_ptm_3_4_ourense.npy")

Y_commute_ptm_3_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/Y_commute_ptm_3_4_ourense.npy")

Z_commute_ptm_3_4 =  np.load(f"./PTM_files/ourense/q4/{encoding}/Z_commute_ptm_3_4_ourense.npy")




ops_ptm_4q = np.zeros((19,3,256,256), dtype = np.complex64)

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

ops_ptm_4q[8,1,:,:] = Z_commute_ptm_2_4

##

ops_ptm_4q[9,0,:,:] = I_ptm_3_4

ops_ptm_4q[9,1,:,:] = X_ptm_3_4

ops_ptm_4q[9,2,:,:] = X_commute_ptm_3_4

ops_ptm_4q[10,0,:,:] = I_ptm_3_4

ops_ptm_4q[10,1,:,:] = Y_ptm_3_4

ops_ptm_4q[10,2,:,:] = Y_commute_ptm_3_4


ops_ptm_4q[11,0,:,:] = I_ptm_3_4

ops_ptm_4q[11,2,:,:] = Z_ptm_3_4

ops_ptm_4q[11,2,:,:] = Z_commute_ptm_3_4

##

ops_ptm_4q[12,0,:,:] = cx_ptm_01_4


#

ops_ptm_4q[13,0,:,:] = cx_ptm_10_4

ops_ptm_4q[14,0,:,:] = cx_ptm_12_4

ops_ptm_4q[15,0,:,:] = cx_ptm_13_4

#


ops_ptm_4q[16,0,:,:] = cx_ptm_21_4

#


ops_ptm_4q[17,0,:,:] = cx_ptm_31_4

ops_ptm_4q[18,0,:,:] = ReadOut_4_ptm

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


