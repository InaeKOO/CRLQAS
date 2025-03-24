import numpy as np
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import H, CNOT, RX
import random

I2 = np.eye(2)
H_mat = (1/np.sqrt(2)) * np.array([[1, 1],
                                [1, -1]])
Rx_neg = (1/np.sqrt(2)) * np.array([[1,  1j],
                                    [1j, 1]])
Rx_pos = (1/np.sqrt(2)) * np.array([[1, -1j],
                                    [-1j, 1]])

def get_random_basis():
    return random.choice(['X', 'Y', 'Z'])

def get_measurement_gate(basis, qubit):
    if basis == 'X':
        return H(qubit)
    elif basis == 'Y':
        return RX(qubit, -np.pi/2)
    else:
        return None
    

def get_reconstruction_unitary(basis):

    if basis == 'X':
        return H_mat
    elif basis == 'Y':
        return Rx_pos
    else:
        return I2
    
def get_single_qubit_estimator(basis, outcome):
    ket0 = np.array([1, 0])
    ket1 = np.array([0, 1])
    ket = ket0 if outcome == 0 else ket1
    U_rec = get_reconstruction_unitary(basis)
    # Rotate back to the original basis.
    v = U_rec @ ket
    rho_est = 3 * np.outer(v, np.conjugate(v)) - I2
    return rho_est

def tensor_list(matrices):
    """Return the tensor (Kronecker) product of a list of matrices."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result

def is_density_matrix(rho, tol=1e-8):
    """
    주어진 행렬 rho가 밀도 행렬인지 확인합니다.
    
    Parameters:
        rho (np.ndarray): 검사할 행렬.
        tol (float): 허용 오차.
    
    Returns:
        bool: 밀도 행렬이면 True, 아니면 False.
    """
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        print("행렬이 정사각행렬이 아님.")
        return False

    if not np.allclose(rho, rho.conj().T, atol=tol):
        print("행렬이 에르미트(켤레 전치) 조건을 만족하지 않음.")
        return False

    eigenvalues = np.linalg.eigvalsh(rho)
    if np.any(eigenvalues < -tol):
        print(eigenvalues)
        print("행렬의 고윳값 중 음수가 존재함.")
        return False

    trace = np.trace(rho)
    if not np.isclose(trace, 1.0, atol=tol):
        print(f"행렬의 추적이 1이 아님 (Tr = {trace}).")
        return False

    return True

if __name__ == '__main__':
    pass