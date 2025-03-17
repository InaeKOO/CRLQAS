import jax.numpy as jnp
from jax import jit
import jax
import numpy as np
from environment.VQE_files.quant.quant_lib_3q_10max import get_m_3q, mv_64, mv_8, get_m_ptm_3q
from environment.VQE_files.VQE_helper import shot_noise_np, final_energy,  final_energy_ptm, get_instr
from functools import partial

def get_instrs_3q(state):
    n_qubits = 3
    num_gates = int(state[:, 0:n_qubits +3 , :].sum())
    if num_gates == 0:
        return np.array([[15,0]]), np.array([])
    circ_instrs_np = np.zeros((num_gates + 1, 2))
    gate_ctr = 0

    for _, local_state in enumerate(state):

        thetas = local_state[n_qubits + 3:]
        rot_pos = (local_state[n_qubits: n_qubits + 3] == 1).nonzero(as_tuple=True)
        cnot_pos = (local_state[:n_qubits] == 1).nonzero(as_tuple=True)

        targ = cnot_pos[1]
        ctrl = cnot_pos[0]

        rot_direction_list = rot_pos[0]
        rot_qubit_list = rot_pos[1]

        if len(ctrl) != 0:
            for r in range(len(ctrl)):
                circ_instrs_np[gate_ctr, 0] = cx_3q_actnum(ctrl[r], targ[r])
                gate_ctr += 1

        if len(rot_qubit_list) != 0:
            for pos, r in enumerate(rot_direction_list):
                rot_qubit = int(rot_qubit_list[pos])
                circ_instrs_np[gate_ctr, 0] = rotate_3q_actnum(rot_qubit, r)

                if r == 0:

                    circ_instrs_np[gate_ctr, 1] = thetas[0][rot_qubit]
                elif r == 1:
                    circ_instrs_np[gate_ctr, 1] = thetas[1][rot_qubit]
                elif r == 2:
                    circ_instrs_np[gate_ctr, 1] = thetas[2][rot_qubit]

                gate_ctr += 1


    circ_instrs_np[-1,0] = 15 
    circ_instrs_np = np.array(circ_instrs_np)

    return circ_instrs_np, np.where(circ_instrs_np[:, 0] < 9)

def cx_3q_actnum(ctrl, targ):
    actnum = 9
    if ctrl == 0:
        if targ == 1:
            actnum += 0
        elif targ == 2:
            actnum += 1
    elif ctrl == 1:
        actnum += 2
        if targ == 0:
            actnum += 0
        elif targ == 2:
            actnum += 1
    elif ctrl == 2:
        actnum += 4
        if targ == 0:
            actnum += 0
        elif targ == 1:
            actnum += 1

    return actnum

def rotate_3q_actnum(rot_qub, rot_dir):
    return 3 * rot_qub + rot_dir



def get_exp_val_static_3q(angles, param_circ):
    exp_val = 0
    if len(param_circ.noise_values) == 0:
        exp_val += get_noiseless_exp_val_static_3q( angles, param_circ)
    else:
        exp_val += get_noisy_exp_val_static_3q( angles, param_circ)

    return exp_val
def get_exp_val_3q(angles, param_circ, sigma):


    exp_val = 0
    if len(param_circ.noise_values) == 0:
        exp_val += get_noiseless_exp_val_3q( angles, param_circ)
    else:
        exp_val += get_noisy_exp_val_3q(angles, param_circ) 

    #


    #
    #
    #

    if (param_circ.Nshots > 0):

        shot_noise = shot_noise_np(param_circ.weights, sigma)
        exp_val += shot_noise

    return exp_val

def get_noiseless_exp_val_static_3q(angles,param_circ):

    circ_instrs_jnp = param_circ.putangles(angles)
    Hamil = param_circ.Hamil
    st = param_circ.st
    energy_shift = param_circ.energy_shift
    num_gates = param_circ.num_gates

    exp_val = get_noiseless_energy_static_3q(st, circ_instrs_jnp, num_gates, Hamil, energy_shift)
    return exp_val

def get_noiseless_exp_val_3q(angles,param_circ):

    circ_instrs_jnp = param_circ.putangles(angles)
    Hamil = param_circ.Hamil
    st = param_circ.st
    energy_shift = param_circ.energy_shift
    num_gates = param_circ.num_gates


    if num_gates == 1:
        exp_val = get_noiseless_energy_1g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 2:
        exp_val = get_noiseless_energy_2g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 3:
        exp_val = get_noiseless_energy_3g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 4:
        exp_val = get_noiseless_energy_4g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 5:
        exp_val = get_noiseless_energy_5g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 6:
        exp_val = get_noiseless_energy_6g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 7:
        exp_val = get_noiseless_energy_7g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 8:
        exp_val = get_noiseless_energy_8g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 9:
        exp_val = get_noiseless_energy_9g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 10:
        exp_val = get_noiseless_energy_10g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 11:
        exp_val = get_noiseless_energy_11g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 12:
        exp_val = get_noiseless_energy_12g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 13:
        exp_val = get_noiseless_energy_13g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 14:
        exp_val = get_noiseless_energy_14g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 15:
        exp_val = get_noiseless_energy_15g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 16:
        exp_val = get_noiseless_energy_16g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 17:
        exp_val = get_noiseless_energy_17g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 18:
        exp_val = get_noiseless_energy_18g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 19:
        exp_val = get_noiseless_energy_19g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 20:
        exp_val = get_noiseless_energy_20g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 21:
        exp_val = get_noiseless_energy_21g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 22:
        exp_val = get_noiseless_energy_22g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 23:
        exp_val = get_noiseless_energy_23g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 24:
        exp_val = get_noiseless_energy_24g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 25:
        exp_val = get_noiseless_energy_25g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 26:
        exp_val = get_noiseless_energy_26g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 27:
        exp_val = get_noiseless_energy_27g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 28:
        exp_val = get_noiseless_energy_28g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 29:
        exp_val = get_noiseless_energy_29g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 30:
        exp_val = get_noiseless_energy_30g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 31:
        exp_val = get_noiseless_energy_31g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 32:
        exp_val = get_noiseless_energy_32g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 33:
        exp_val = get_noiseless_energy_33g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 34:
        exp_val = get_noiseless_energy_34g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 35:
        exp_val = get_noiseless_energy_35g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 36:
        exp_val = get_noiseless_energy_36g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 37:
        exp_val = get_noiseless_energy_37g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 38:
        exp_val = get_noiseless_energy_38g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 39:
        exp_val = get_noiseless_energy_39g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 40:
        exp_val = get_noiseless_energy_40g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 41:
        exp_val = get_noiseless_energy_41g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 42:
        exp_val = get_noiseless_energy_42g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 43:
        exp_val = get_noiseless_energy_43g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 44:
        exp_val = get_noiseless_energy_44g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 45:
        exp_val = get_noiseless_energy_45g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 46:
        exp_val = get_noiseless_energy_46g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 47:
        exp_val = get_noiseless_energy_47g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 48:
        exp_val = get_noiseless_energy_48g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 49:
        exp_val = get_noiseless_energy_49g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 50:
        exp_val = get_noiseless_energy_50g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 51:
        exp_val = get_noiseless_energy_51g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 52:
        exp_val = get_noiseless_energy_52g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 53:
        exp_val = get_noiseless_energy_53g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 54:
        exp_val = get_noiseless_energy_54g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 55:
        exp_val = get_noiseless_energy_55g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 56:
        exp_val = get_noiseless_energy_56g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 57:
        exp_val = get_noiseless_energy_57g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 58:
        exp_val = get_noiseless_energy_58g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 59:
        exp_val = get_noiseless_energy_59g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 60:
        exp_val = get_noiseless_energy_60g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 61:
        exp_val = get_noiseless_energy_61g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 62:
        exp_val = get_noiseless_energy_62g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 63:
        exp_val = get_noiseless_energy_63g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 64:
        exp_val = get_noiseless_energy_64g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 65:
        exp_val = get_noiseless_energy_65g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 66:
        exp_val = get_noiseless_energy_66g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 67:
        exp_val = get_noiseless_energy_67g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 68:
        exp_val = get_noiseless_energy_68g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 69:
        exp_val = get_noiseless_energy_69g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 70:
        exp_val = get_noiseless_energy_70g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 71:
        exp_val = get_noiseless_energy_71g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 72:
        exp_val = get_noiseless_energy_72g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 73:
        exp_val = get_noiseless_energy_73g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 74:
        exp_val = get_noiseless_energy_74g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 75:
        exp_val = get_noiseless_energy_75g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 76:
        exp_val = get_noiseless_energy_76g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 77:
        exp_val = get_noiseless_energy_77g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 78:
        exp_val = get_noiseless_energy_78g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 79:
        exp_val = get_noiseless_energy_79g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 80:
        exp_val = get_noiseless_energy_80g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 81:
        exp_val = get_noiseless_energy_81g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 82:
        exp_val = get_noiseless_energy_82g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 83:
        exp_val = get_noiseless_energy_83g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 84:
        exp_val = get_noiseless_energy_84g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 85:
        exp_val = get_noiseless_energy_85g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 86:
        exp_val = get_noiseless_energy_86g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 87:
        exp_val = get_noiseless_energy_87g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 88:
        exp_val = get_noiseless_energy_88g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 89:
        exp_val = get_noiseless_energy_89g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 90:
        exp_val = get_noiseless_energy_90g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 91:
        exp_val = get_noiseless_energy_91g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 92:
        exp_val = get_noiseless_energy_92g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 93:
        exp_val = get_noiseless_energy_93g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 94:
        exp_val = get_noiseless_energy_94g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 95:
        exp_val = get_noiseless_energy_95g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 96:
        exp_val = get_noiseless_energy_96g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 97:
        exp_val = get_noiseless_energy_97g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 98:
        exp_val = get_noiseless_energy_98g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 99:
        exp_val = get_noiseless_energy_99g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 100:
        exp_val = get_noiseless_energy_100g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 101:
        exp_val = get_noiseless_energy_101g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 102:
        exp_val = get_noiseless_energy_102g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 103:
        exp_val = get_noiseless_energy_103g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 104:
        exp_val = get_noiseless_energy_104g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 105:
        exp_val = get_noiseless_energy_105g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 106:
        exp_val = get_noiseless_energy_106g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 107:
        exp_val = get_noiseless_energy_107g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 108:
        exp_val = get_noiseless_energy_108g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 109:
        exp_val = get_noiseless_energy_109g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 110:
        exp_val = get_noiseless_energy_110g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 111:
        exp_val = get_noiseless_energy_111g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 112:
        exp_val = get_noiseless_energy_112g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 113:
        exp_val = get_noiseless_energy_113g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 114:
        exp_val = get_noiseless_energy_114g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 115:
        exp_val = get_noiseless_energy_115g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 116:
        exp_val = get_noiseless_energy_116g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 117:
        exp_val = get_noiseless_energy_117g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 118:
        exp_val = get_noiseless_energy_118g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 119:
        exp_val = get_noiseless_energy_119g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 120:
        exp_val = get_noiseless_energy_120g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 121:
        exp_val = get_noiseless_energy_121g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 122:
        exp_val = get_noiseless_energy_122g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 123:
        exp_val = get_noiseless_energy_123g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 124:
        exp_val = get_noiseless_energy_124g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 125:
        exp_val = get_noiseless_energy_125g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 126:
        exp_val = get_noiseless_energy_126g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 127:
        exp_val = get_noiseless_energy_127g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 128:
        exp_val = get_noiseless_energy_128g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 129:
        exp_val = get_noiseless_energy_129g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 130:
        exp_val = get_noiseless_energy_130g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 131:
        exp_val = get_noiseless_energy_131g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 132:
        exp_val = get_noiseless_energy_132g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 133:
        exp_val = get_noiseless_energy_133g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 134:
        exp_val = get_noiseless_energy_134g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 135:
        exp_val = get_noiseless_energy_135g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 136:
        exp_val = get_noiseless_energy_136g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 137:
        exp_val = get_noiseless_energy_137g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 138:
        exp_val = get_noiseless_energy_138g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 139:
        exp_val = get_noiseless_energy_139g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 140:
        exp_val = get_noiseless_energy_140g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 141:
        exp_val = get_noiseless_energy_141g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 142:
        exp_val = get_noiseless_energy_142g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 143:
        exp_val = get_noiseless_energy_143g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 144:
        exp_val = get_noiseless_energy_144g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 145:
        exp_val = get_noiseless_energy_145g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 146:
        exp_val = get_noiseless_energy_146g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 147:
        exp_val = get_noiseless_energy_147g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 148:
        exp_val = get_noiseless_energy_148g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 149:
        exp_val = get_noiseless_energy_149g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 150:
        exp_val = get_noiseless_energy_150g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 151:
        exp_val = get_noiseless_energy_151g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 152:
        exp_val = get_noiseless_energy_152g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 153:
        exp_val = get_noiseless_energy_153g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 154:
        exp_val = get_noiseless_energy_154g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 155:
        exp_val = get_noiseless_energy_155g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 156:
        exp_val = get_noiseless_energy_156g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 157:
        exp_val = get_noiseless_energy_157g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 158:
        exp_val = get_noiseless_energy_158g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 159:
        exp_val = get_noiseless_energy_159g_3q(st, circ_instrs_jnp, Hamil, energy_shift)


    elif num_gates == 160:
        exp_val = get_noiseless_energy_160g_3q(st, circ_instrs_jnp, Hamil, energy_shift)

    return exp_val


def get_noisy_exp_val_static_3q(angles,param_circ):

    circ_instrs_jnp = param_circ.putangles(angles)
    Hamil_ptm = param_circ.Hamil_ptm
    rho_ptm = param_circ.rho_ptm
    energy_shift = param_circ.energy_shift
    num_gates = param_circ.num_gates

    exp_val = get_noisy_energy_static_3q(rho_ptm, circ_instrs_jnp, num_gates, Hamil_ptm, energy_shift)
    return exp_val

def get_noisy_exp_val_3q(angles,param_circ):
    circ_instrs_jnp = param_circ.putangles(angles)
    Hamil_ptm = param_circ.Hamil_ptm
    rho_ptm = param_circ.rho_ptm
    energy_shift = param_circ.energy_shift
    num_gates = param_circ.num_gates


    if num_gates == 1:
        exp_val = get_noisy_energy_1g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 2:
        exp_val = get_noisy_energy_2g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 3:
        exp_val = get_noisy_energy_3g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 4:
        exp_val = get_noisy_energy_4g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 5:
        exp_val = get_noisy_energy_5g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 6:
        exp_val = get_noisy_energy_6g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 7:
        exp_val = get_noisy_energy_7g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 8:
        exp_val = get_noisy_energy_8g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 9:
        exp_val = get_noisy_energy_9g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 10:
        exp_val = get_noisy_energy_10g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 11:
        exp_val = get_noisy_energy_11g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 12:
        exp_val = get_noisy_energy_12g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 13:
        exp_val = get_noisy_energy_13g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 14:
        exp_val = get_noisy_energy_14g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 15:
        exp_val = get_noisy_energy_15g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 16:
        exp_val = get_noisy_energy_16g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 17:
        exp_val = get_noisy_energy_17g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 18:
        exp_val = get_noisy_energy_18g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 19:
        exp_val = get_noisy_energy_19g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 20:
        exp_val = get_noisy_energy_20g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 21:
        exp_val = get_noisy_energy_21g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 22:
        exp_val = get_noisy_energy_22g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 23:
        exp_val = get_noisy_energy_23g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 24:
        exp_val = get_noisy_energy_24g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 25:
        exp_val = get_noisy_energy_25g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 26:
        exp_val = get_noisy_energy_26g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 27:
        exp_val = get_noisy_energy_27g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 28:
        exp_val = get_noisy_energy_28g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 29:
        exp_val = get_noisy_energy_29g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 30:
        exp_val = get_noisy_energy_30g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 31:
        exp_val = get_noisy_energy_31g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 32:
        exp_val = get_noisy_energy_32g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 33:
        exp_val = get_noisy_energy_33g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 34:
        exp_val = get_noisy_energy_34g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 35:
        exp_val = get_noisy_energy_35g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 36:
        exp_val = get_noisy_energy_36g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 37:
        exp_val = get_noisy_energy_37g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 38:
        exp_val = get_noisy_energy_38g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 39:
        exp_val = get_noisy_energy_39g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 40:
        exp_val = get_noisy_energy_40g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 41:
        exp_val = get_noisy_energy_41g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 42:
        exp_val = get_noisy_energy_42g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 43:
        exp_val = get_noisy_energy_43g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 44:
        exp_val = get_noisy_energy_44g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 45:
        exp_val = get_noisy_energy_45g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 46:
        exp_val = get_noisy_energy_46g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 47:
        exp_val = get_noisy_energy_47g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 48:
        exp_val = get_noisy_energy_48g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 49:
        exp_val = get_noisy_energy_49g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 50:
        exp_val = get_noisy_energy_50g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 51:
        exp_val = get_noisy_energy_51g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 52:
        exp_val = get_noisy_energy_52g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 53:
        exp_val = get_noisy_energy_53g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 54:
        exp_val = get_noisy_energy_54g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 55:
        exp_val = get_noisy_energy_55g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 56:
        exp_val = get_noisy_energy_56g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 57:
        exp_val = get_noisy_energy_57g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 58:
        exp_val = get_noisy_energy_58g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 59:
        exp_val = get_noisy_energy_59g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 60:
        exp_val = get_noisy_energy_60g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 61:
        exp_val = get_noisy_energy_61g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 62:
        exp_val = get_noisy_energy_62g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 63:
        exp_val = get_noisy_energy_63g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 64:
        exp_val = get_noisy_energy_64g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 65:
        exp_val = get_noisy_energy_65g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 66:
        exp_val = get_noisy_energy_66g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 67:
        exp_val = get_noisy_energy_67g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 68:
        exp_val = get_noisy_energy_68g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 69:
        exp_val = get_noisy_energy_69g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 70:
        exp_val = get_noisy_energy_70g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 71:
        exp_val = get_noisy_energy_71g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 72:
        exp_val = get_noisy_energy_72g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 73:
        exp_val = get_noisy_energy_73g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 74:
        exp_val = get_noisy_energy_74g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 75:
        exp_val = get_noisy_energy_75g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 76:
        exp_val = get_noisy_energy_76g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 77:
        exp_val = get_noisy_energy_77g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 78:
        exp_val = get_noisy_energy_78g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 79:
        exp_val = get_noisy_energy_79g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 80:
        exp_val = get_noisy_energy_80g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 81:
        exp_val = get_noisy_energy_81g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 82:
        exp_val = get_noisy_energy_82g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 83:
        exp_val = get_noisy_energy_83g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 84:
        exp_val = get_noisy_energy_84g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 85:
        exp_val = get_noisy_energy_85g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 86:
        exp_val = get_noisy_energy_86g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 87:
        exp_val = get_noisy_energy_87g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 88:
        exp_val = get_noisy_energy_88g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 89:
        exp_val = get_noisy_energy_89g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 90:
        exp_val = get_noisy_energy_90g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 91:
        exp_val = get_noisy_energy_91g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 92:
        exp_val = get_noisy_energy_92g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 93:
        exp_val = get_noisy_energy_93g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 94:
        exp_val = get_noisy_energy_94g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 95:
        exp_val = get_noisy_energy_95g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 96:
        exp_val = get_noisy_energy_96g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 97:
        exp_val = get_noisy_energy_97g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 98:
        exp_val = get_noisy_energy_98g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 99:
        exp_val = get_noisy_energy_99g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 100:
        exp_val = get_noisy_energy_100g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 101:
        exp_val = get_noisy_energy_101g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 102:
        exp_val = get_noisy_energy_102g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 103:
        exp_val = get_noisy_energy_103g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 104:
        exp_val = get_noisy_energy_104g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 105:
        exp_val = get_noisy_energy_105g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 106:
        exp_val = get_noisy_energy_106g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 107:
        exp_val = get_noisy_energy_107g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 108:
        exp_val = get_noisy_energy_108g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 109:
        exp_val = get_noisy_energy_109g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 110:
        exp_val = get_noisy_energy_110g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 111:
        exp_val = get_noisy_energy_111g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 112:
        exp_val = get_noisy_energy_112g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 113:
        exp_val = get_noisy_energy_113g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 114:
        exp_val = get_noisy_energy_114g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 115:
        exp_val = get_noisy_energy_115g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 116:
        exp_val = get_noisy_energy_116g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 117:
        exp_val = get_noisy_energy_117g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 118:
        exp_val = get_noisy_energy_118g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 119:
        exp_val = get_noisy_energy_119g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 120:
        exp_val = get_noisy_energy_120g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 121:
        exp_val = get_noisy_energy_121g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 122:
        exp_val = get_noisy_energy_122g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 123:
        exp_val = get_noisy_energy_123g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 124:
        exp_val = get_noisy_energy_124g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 125:
        exp_val = get_noisy_energy_125g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 126:
        exp_val = get_noisy_energy_126g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 127:
        exp_val = get_noisy_energy_127g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 128:
        exp_val = get_noisy_energy_128g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 129:
        exp_val = get_noisy_energy_129g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 130:
        exp_val = get_noisy_energy_130g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 131:
        exp_val = get_noisy_energy_131g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 132:
        exp_val = get_noisy_energy_132g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 133:
        exp_val = get_noisy_energy_133g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 134:
        exp_val = get_noisy_energy_134g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 135:
        exp_val = get_noisy_energy_135g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 136:
        exp_val = get_noisy_energy_136g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 137:
        exp_val = get_noisy_energy_137g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 138:
        exp_val = get_noisy_energy_138g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 139:
        exp_val = get_noisy_energy_139g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 140:
        exp_val = get_noisy_energy_140g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 141:
        exp_val = get_noisy_energy_141g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 142:
        exp_val = get_noisy_energy_142g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 143:
        exp_val = get_noisy_energy_143g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 144:
        exp_val = get_noisy_energy_144g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 145:
        exp_val = get_noisy_energy_145g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 146:
        exp_val = get_noisy_energy_146g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 147:
        exp_val = get_noisy_energy_147g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 148:
        exp_val = get_noisy_energy_148g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 149:
        exp_val = get_noisy_energy_149g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 150:
        exp_val = get_noisy_energy_150g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 151:
        exp_val = get_noisy_energy_151g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 152:
        exp_val = get_noisy_energy_152g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 153:
        exp_val = get_noisy_energy_153g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 154:
        exp_val = get_noisy_energy_154g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 155:
        exp_val = get_noisy_energy_155g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 156:
        exp_val = get_noisy_energy_156g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 157:
        exp_val = get_noisy_energy_157g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 158:
        exp_val = get_noisy_energy_158g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 159:
        exp_val = get_noisy_energy_159g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)


    elif num_gates == 160:
        exp_val = get_noisy_energy_160g_3q(rho_ptm, circ_instrs_jnp, Hamil_ptm, energy_shift)

    return exp_val


@jit
def body_fun_instr_noiseless_3q(iq, st, circ_instrs_jnp):
    opnum, opang = get_instr(circ_instrs_jnp[iq, :])
    m = get_m_3q(opnum, opang)

    st = mv_8(m, st)

    return st

@partial(jit , static_argnums = (2))
def get_st_aft_circ_noiseless_static_3q(st_b4, circ_instrs_jnp, num_gates):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, num_gates, layer_body_fn, st_after)
    return st_after

@partial(jit , static_argnums = (2))
def get_noiseless_energy_static_3q(st_b4, circ_instrs_jnp, num_gates, Hamil, energy_shift):
    st = get_st_aft_circ_noiseless_static_3q(st_b4, circ_instrs_jnp, num_gates)
    return final_energy(st, Hamil, energy_shift)
@jit
def body_fun_instr_ptm_noisy_3q(iq, rho_ptm, circ_instrs_jnp):
    opnum, opang = get_instr(circ_instrs_jnp[iq, :])
    m = get_m_ptm_3q(opnum, opang)

    rho_ptm = mv_64(m, rho_ptm)

    return rho_ptm

@partial(jit , static_argnums = (2))
def get_rho_ptm_aft_circ_noisy_static_3q(rho_ptm_b4, circ_instrs_jnp, num_gates):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, num_gates, layer_body_fn, rho_ptm_after)
    return rho_ptm_after

@partial(jit , static_argnums = (2))
def get_noisy_energy_static_3q(rho_ptm_b4, circ_instrs_jnp,num_gates, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_static_3q(rho_ptm_b4, circ_instrs_jnp,num_gates)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_1g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 1, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_1g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_1g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_2g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 2, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_2g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_2g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_3g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 3, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_3g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_3g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_4g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 4, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_4g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_4g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_5g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 5, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_5g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_5g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_6g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 6, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_6g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_6g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_7g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 7, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_7g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_7g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_8g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 8, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_8g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_8g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_9g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 9, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_9g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_9g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_10g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 10, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_10g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_10g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_11g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 11, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_11g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_11g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_12g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 12, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_12g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_12g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_13g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 13, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_13g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_13g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_14g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 14, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_14g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_14g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_15g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 15, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_15g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_15g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_16g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 16, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_16g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_16g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_17g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 17, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_17g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_17g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_18g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 18, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_18g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_18g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_19g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 19, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_19g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_19g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_20g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 20, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_20g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_20g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_21g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 21, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_21g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_21g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_22g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 22, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_22g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_22g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_23g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 23, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_23g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_23g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_24g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 24, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_24g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_24g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_25g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 25, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_25g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_25g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_26g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 26, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_26g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_26g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_27g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 27, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_27g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_27g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_28g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 28, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_28g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_28g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_29g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 29, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_29g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_29g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_30g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 30, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_30g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_30g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_31g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 31, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_31g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_31g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_32g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 32, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_32g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_32g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_33g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 33, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_33g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_33g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_34g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 34, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_34g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_34g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_35g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 35, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_35g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_35g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_36g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 36, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_36g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_36g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_37g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 37, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_37g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_37g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_38g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 38, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_38g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_38g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_39g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 39, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_39g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_39g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_40g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 40, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_40g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_40g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_41g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 41, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_41g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_41g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_42g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 42, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_42g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_42g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_43g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 43, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_43g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_43g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_44g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 44, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_44g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_44g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_45g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 45, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_45g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_45g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_46g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 46, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_46g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_46g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_47g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 47, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_47g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_47g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_48g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 48, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_48g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_48g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_49g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 49, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_49g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_49g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_50g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 50, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_50g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_50g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_51g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 51, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_51g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_51g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_52g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 52, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_52g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_52g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_53g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 53, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_53g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_53g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_54g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 54, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_54g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_54g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_55g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 55, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_55g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_55g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_56g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 56, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_56g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_56g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_57g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 57, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_57g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_57g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_58g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 58, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_58g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_58g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_59g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 59, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_59g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_59g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_60g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 60, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_60g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_60g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_61g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 61, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_61g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_61g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_62g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 62, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_62g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_62g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_63g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 63, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_63g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_63g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_64g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 64, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_64g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_64g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_65g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 65, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_65g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_65g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_66g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 66, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_66g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_66g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_67g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 67, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_67g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_67g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_68g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 68, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_68g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_68g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_69g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 69, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_69g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_69g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_70g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 70, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_70g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_70g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_71g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 71, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_71g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_71g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_72g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 72, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_72g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_72g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_73g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 73, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_73g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_73g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_74g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 74, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_74g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_74g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_75g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 75, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_75g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_75g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_76g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 76, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_76g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_76g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_77g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 77, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_77g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_77g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_78g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 78, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_78g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_78g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_79g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 79, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_79g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_79g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_80g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 80, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_80g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_80g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_81g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 81, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_81g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_81g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_82g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 82, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_82g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_82g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_83g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 83, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_83g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_83g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_84g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 84, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_84g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_84g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_85g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 85, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_85g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_85g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_86g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 86, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_86g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_86g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_87g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 87, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_87g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_87g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_88g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 88, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_88g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_88g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_89g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 89, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_89g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_89g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_90g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 90, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_90g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_90g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_91g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 91, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_91g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_91g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_92g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 92, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_92g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_92g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_93g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 93, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_93g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_93g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_94g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 94, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_94g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_94g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_95g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 95, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_95g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_95g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_96g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 96, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_96g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_96g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_97g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 97, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_97g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_97g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_98g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 98, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_98g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_98g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_99g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 99, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_99g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_99g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_100g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 100, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_100g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_100g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_101g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 101, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_101g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_101g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_102g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 102, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_102g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_102g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_103g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 103, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_103g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_103g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_104g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 104, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_104g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_104g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_105g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 105, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_105g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_105g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_106g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 106, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_106g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_106g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_107g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 107, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_107g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_107g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_108g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 108, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_108g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_108g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_109g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 109, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_109g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_109g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_110g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 110, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_110g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_110g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_111g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 111, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_111g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_111g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_112g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 112, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_112g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_112g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_113g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 113, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_113g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_113g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_114g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 114, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_114g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_114g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_115g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 115, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_115g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_115g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_116g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 116, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_116g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_116g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_117g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 117, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_117g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_117g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_118g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 118, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_118g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_118g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_119g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 119, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_119g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_119g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_120g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 120, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_120g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_120g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_121g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 121, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_121g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_121g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_122g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 122, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_122g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_122g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_123g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 123, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_123g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_123g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_124g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 124, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_124g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_124g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_125g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 125, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_125g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_125g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_126g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 126, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_126g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_126g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_127g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 127, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_127g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_127g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_128g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 128, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_128g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_128g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_129g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 129, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_129g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_129g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_130g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 130, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_130g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_130g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_131g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 131, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_131g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_131g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_132g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 132, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_132g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_132g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_133g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 133, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_133g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_133g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_134g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 134, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_134g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_134g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_135g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 135, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_135g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_135g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_136g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 136, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_136g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_136g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_137g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 137, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_137g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_137g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_138g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 138, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_138g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_138g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_139g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 139, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_139g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_139g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_140g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 140, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_140g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_140g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_141g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 141, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_141g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_141g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_142g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 142, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_142g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_142g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_143g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 143, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_143g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_143g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_144g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 144, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_144g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_144g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_145g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 145, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_145g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_145g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_146g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 146, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_146g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_146g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_147g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 147, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_147g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_147g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_148g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 148, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_148g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_148g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_149g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 149, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_149g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_149g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_150g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 150, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_150g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_150g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_151g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 151, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_151g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_151g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_152g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 152, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_152g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_152g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_153g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 153, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_153g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_153g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_154g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 154, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_154g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_154g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_155g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 155, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_155g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_155g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_156g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 156, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_156g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_156g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_157g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 157, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_157g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_157g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_158g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 158, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_158g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_158g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_159g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 159, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_159g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_159g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)


@jax.jit
def get_rho_ptm_aft_circ_noisy_160g_3q(rho_ptm_b4, circ_instrs_jnp):
    rho_ptm_after = rho_ptm_b4
    layer_body_fn = partial(body_fun_instr_ptm_noisy_3q, circ_instrs_jnp=circ_instrs_jnp)
    rho_ptm_after = jax.lax.fori_loop(0, 160, layer_body_fn, rho_ptm_after)
    return rho_ptm_after


@jax.jit
def get_noisy_energy_160g_3q(rho_ptm_b4, circ_instrs_jnp, Hamil_ptm, energy_shift):
    rho_ptm = get_rho_ptm_aft_circ_noisy_160g_3q(rho_ptm_b4, circ_instrs_jnp)
    return final_energy_ptm(rho_ptm, Hamil_ptm, energy_shift)




@jax.jit
def get_st_aft_noiseless_1g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 1, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_1g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_1g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_2g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 2, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_2g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_2g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_3g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 3, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_3g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_3g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_4g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 4, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_4g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_4g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_5g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 5, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_5g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_5g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_6g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 6, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_6g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_6g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_7g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 7, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_7g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_7g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_8g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 8, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_8g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_8g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_9g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 9, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_9g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_9g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_10g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 10, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_10g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_10g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_11g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 11, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_11g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_11g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_12g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 12, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_12g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_12g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_13g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 13, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_13g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_13g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_14g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 14, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_14g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_14g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_15g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 15, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_15g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_15g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_16g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 16, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_16g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_16g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_17g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 17, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_17g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_17g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_18g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 18, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_18g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_18g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_19g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 19, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_19g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_19g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_20g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 20, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_20g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_20g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_21g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 21, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_21g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_21g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_22g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 22, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_22g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_22g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_23g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 23, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_23g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_23g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_24g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 24, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_24g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_24g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_25g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 25, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_25g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_25g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_26g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 26, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_26g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_26g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_27g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 27, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_27g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_27g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_28g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 28, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_28g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_28g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_29g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 29, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_29g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_29g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_30g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 30, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_30g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_30g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_31g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 31, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_31g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_31g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_32g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 32, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_32g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_32g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_33g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 33, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_33g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_33g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_34g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 34, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_34g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_34g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_35g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 35, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_35g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_35g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_36g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 36, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_36g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_36g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_37g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 37, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_37g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_37g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_38g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 38, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_38g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_38g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_39g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 39, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_39g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_39g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_40g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 40, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_40g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_40g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_41g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 41, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_41g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_41g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_42g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 42, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_42g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_42g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_43g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 43, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_43g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_43g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_44g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 44, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_44g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_44g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_45g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 45, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_45g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_45g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_46g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 46, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_46g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_46g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_47g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 47, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_47g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_47g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_48g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 48, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_48g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_48g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_49g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 49, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_49g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_49g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_50g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 50, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_50g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_50g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_51g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 51, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_51g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_51g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_52g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 52, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_52g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_52g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_53g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 53, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_53g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_53g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_54g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 54, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_54g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_54g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_55g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 55, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_55g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_55g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_56g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 56, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_56g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_56g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_57g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 57, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_57g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_57g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_58g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 58, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_58g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_58g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_59g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 59, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_59g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_59g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_60g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 60, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_60g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_60g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_61g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 61, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_61g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_61g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_62g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 62, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_62g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_62g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_63g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 63, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_63g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_63g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_64g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 64, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_64g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_64g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_65g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 65, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_65g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_65g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_66g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 66, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_66g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_66g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_67g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 67, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_67g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_67g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_68g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 68, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_68g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_68g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_69g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 69, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_69g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_69g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_70g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 70, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_70g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_70g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_71g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 71, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_71g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_71g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_72g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 72, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_72g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_72g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_73g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 73, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_73g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_73g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_74g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 74, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_74g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_74g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_75g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 75, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_75g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_75g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_76g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 76, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_76g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_76g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_77g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 77, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_77g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_77g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_78g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 78, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_78g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_78g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_79g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 79, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_79g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_79g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_80g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 80, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_80g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_80g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_81g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 81, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_81g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_81g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_82g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 82, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_82g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_82g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_83g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 83, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_83g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_83g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_84g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 84, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_84g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_84g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_85g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 85, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_85g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_85g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_86g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 86, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_86g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_86g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_87g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 87, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_87g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_87g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_88g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 88, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_88g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_88g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_89g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 89, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_89g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_89g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_90g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 90, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_90g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_90g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_91g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 91, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_91g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_91g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_92g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 92, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_92g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_92g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_93g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 93, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_93g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_93g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_94g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 94, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_94g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_94g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_95g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 95, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_95g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_95g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_96g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 96, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_96g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_96g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_97g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 97, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_97g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_97g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_98g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 98, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_98g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_98g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_99g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 99, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_99g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_99g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_100g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 100, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_100g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_100g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_101g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 101, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_101g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_101g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_102g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 102, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_102g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_102g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_103g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 103, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_103g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_103g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_104g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 104, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_104g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_104g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_105g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 105, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_105g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_105g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_106g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 106, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_106g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_106g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_107g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 107, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_107g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_107g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_108g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 108, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_108g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_108g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_109g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 109, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_109g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_109g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_110g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 110, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_110g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_110g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_111g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 111, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_111g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_111g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_112g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 112, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_112g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_112g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_113g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 113, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_113g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_113g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_114g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 114, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_114g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_114g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_115g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 115, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_115g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_115g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_116g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 116, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_116g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_116g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_117g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 117, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_117g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_117g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_118g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 118, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_118g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_118g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_119g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 119, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_119g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_119g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_120g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 120, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_120g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_120g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_121g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 121, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_121g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_121g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_122g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 122, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_122g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_122g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_123g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 123, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_123g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_123g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_124g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 124, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_124g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_124g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_125g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 125, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_125g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_125g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_126g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 126, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_126g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_126g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_127g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 127, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_127g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_127g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_128g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 128, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_128g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_128g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_129g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 129, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_129g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_129g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_130g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 130, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_130g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_130g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_131g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 131, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_131g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_131g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_132g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 132, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_132g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_132g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_133g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 133, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_133g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_133g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_134g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 134, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_134g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_134g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_135g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 135, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_135g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_135g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_136g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 136, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_136g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_136g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_137g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 137, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_137g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_137g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_138g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 138, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_138g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_138g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_139g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 139, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_139g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_139g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_140g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 140, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_140g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_140g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_141g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 141, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_141g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_141g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_142g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 142, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_142g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_142g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_143g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 143, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_143g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_143g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_144g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 144, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_144g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_144g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_145g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 145, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_145g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_145g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_146g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 146, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_146g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_146g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_147g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 147, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_147g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_147g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_148g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 148, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_148g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_148g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_149g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 149, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_149g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_149g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_150g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 150, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_150g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_150g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_151g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 151, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_151g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_151g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_152g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 152, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_152g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_152g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_153g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 153, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_153g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_153g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_154g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 154, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_154g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_154g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_155g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 155, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_155g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_155g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_156g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 156, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_156g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_156g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_157g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 157, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_157g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_157g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_158g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 158, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_158g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_158g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_159g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 159, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_159g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_159g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)


@jax.jit
def get_st_aft_noiseless_160g_3q(st_b4, circ_instrs_jnp):
    st_after = st_b4
    layer_body_fn = partial(body_fun_instr_noiseless_3q, circ_instrs_jnp=circ_instrs_jnp)
    st_after = jax.lax.fori_loop(0, 160, layer_body_fn, st_after)
    return st_after


@jax.jit
def get_noiseless_energy_160g_3q(st_b4, circ_instrs_jnp, Hamil, energy_shift):
    st = get_st_aft_noiseless_160g_3q(st_b4, circ_instrs_jnp)
    return final_energy(st, Hamil, energy_shift)
