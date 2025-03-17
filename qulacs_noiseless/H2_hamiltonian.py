from openfermion.transforms import *
from openfermion.chem import MolecularData
from openfermion.transforms import binary_code_transform
from openfermion.transforms import get_fermion_operator
from openfermion.linalg import get_sparse_operator
from openfermion.transforms import normal_ordered, reorder
from openfermion.utils import up_then_down
from openfermionpsi4 import run_psi4
from qulacs.observable import create_observable_from_openfermion_text
from dataclasses import dataclass
from scipy.linalg import eig
import numpy as np

def H2_hamiltonian(bond_length, run_scf, run_mp2, run_cisd, run_fci):
    geometry = [('H', (0., 0., 0)), ('H', (0., 0., bond_length))]
    molecule = MolecularData(geometry, 'sto-3g', 1, 0,
                             description=f"{bond_length}")
                             
    
    molecule = run_psi4(molecule,
                    run_scf=run_scf,
                    run_mp2=run_mp2,
                    run_cisd=run_cisd,
                    run_ccsd=run_ccsd,
                    run_fci=run_fci)
    molecule.save()
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()
    hamiltonian = normal_ordered(get_fermion_operator(molecular_hamiltonian))
    print(hamiltonian)
    return hamiltonian, molecule

if __name__ == '__main__':

    bond_length = 0.7414
    run_scf, run_mp2, run_cisd, run_ccsd, run_fci  = 1, 1, 1, 1, 1
        
    hamiltonian, _ = H2_hamiltonian(bond_length, run_scf, run_mp2, run_cisd, run_fci)
    @dataclass
    class Config:
        num_qubits = 3
        problem={"ham_type" : 'H2',
        "geometry" : 'H .0 .0 .0; H .0 .0 0.7414',
        "taper" : 0,
        "mapping" : 'jordan_wigner'}

    __ham = dict()

    if Config.num_qubits == 2:
        jw_H2_2q = binary_code_transform(reorder(hamiltonian,up_then_down), 2*checksum_code(2,1))
        dense_ham = get_sparse_operator(jw_H2_2q).todense()
        energy_shift = 0
        observable = create_observable_from_openfermion_text(str(jw_H2_2q))
        weights = []
        for term_idx in range(observable.get_term_count()):
            weights.append(observable.get_term(term_idx).get_coef())
        eigs = sorted(eig(dense_ham)[0].real)

    if Config.num_qubits == 3:
        jw_H2_3q = binary_code_transform(hamiltonian, checksum_code(4,0))
        dense_ham = get_sparse_operator(jw_H2_3q).todense()
        observable = create_observable_from_openfermion_text(str(jw_H2_3q))
        weights = []
        for term_idx in range(observable.get_term_count()):
            weights.append(observable.get_term(term_idx).get_coef())
        eigs = sorted(eig(dense_ham)[0].real)

    if Config.num_qubits == 4:
        jw_H2_4q = binary_code_transform(hamiltonian, jordan_wigner_code(4))
        dense_ham = get_sparse_operator(jw_H2_4q).todense()
        observable = create_observable_from_openfermion_text(str(jw_H2_4q))
        weights = []
        for term_idx in range(observable.get_term_count()):
            weights.append(observable.get_term(term_idx).get_coef())
        eigs = sorted(eig(dense_ham)[0].real)


    __ham['hamiltonian'],__ham['weights'], __ham['eigvals'], __ham['energy_shift']  = dense_ham, weights, eigs, 0

    __geometry = Config.problem['geometry'].replace(" ", "_")
    print(__ham)
    