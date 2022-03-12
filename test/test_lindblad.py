import numpy as np
import matplotlib.pyplot as plt
import qutip
import tenpy
import unittest

from lindblad import *


def test_dissipator_eigvals(N):
    # Construct the Lindblad superoperator with the dissipator part only
    # N = 2
    RM_D = np.array(qutip.rand_dm_ginibre((N**2-1), rank=None))
    RM_H = np.zeros((N,N))
    alpha, gamma = 0, 1

    # Compute the eigenvalues of the matrix that represents this superoperator in the HS base
    dissipatorpart_eigvals = np.linalg.eigvals(Lindbladian_matrix(N,RM_D,RM_H,alpha,gamma))

    # Construct a dictionary that will contain the zero and notzero eigenvalues of the superoperator
    eigvals_dict = {'zero':[], 'real_negative':[], 'cc_pairs':[]}

    # For eigenvalues that have null real and imaginary part append them in the dictionary at the 
    # 'zero' key, otherwhise append them at the 'real_negative' key and 'cc_pairs' key
    for eigval in dissipatorpart_eigvals:
        if np.abs(np.real(eigval)) < 1e-10 and np.abs(np.imag(eigval)) < 1e-10:
            eigvals_dict['zero'].append(eigval)
        if np.abs(np.real(eigval)) > 1e-7:
            if np.abs(np.imag(eigval)) < 1e-10:
                pass
            else:
                eigvals_dict['cc_pairs'].append(eigval)
        if np.abs(np.imag(eigval)) < 1e-10:
            if np.abs(np.real(eigval)) < 1e-10:
                pass
            else:
                assert np.sign(np.real(eigval)) == -1
                eigvals_dict['real_negative'].append(eigval)

    # Check if there exist a single null eigenvalue
    assert len(eigvals_dict['zero']) == 1

    # In order to check if the not zero eigenvalues are in complex and conjugate pairs, construct a
    # new list with these eigenvalues, rounded to 10 decimals and taken in absolute value, and from
    # it construct a set, so a list of unique elements. Finally check that each element in this set
    # appears two times in the initial list.
    abs_ccpairs_eigvals = np.round(np.abs(eigvals_dict['cc_pairs']),10)
    set_ccpairs_eigvals = set(abs_ccpairs_eigvals)

    if N != 2:
        for item in set_ccpairs_eigvals:
            assert list(abs_ccpairs_eigvals).count(item) == 2
    
    # Check if the sum of the eigenvalues is N**2
    sum_eigvals = 0
    for key in eigvals_dict.keys():
        if eigvals_dict[f'{key}']:  # implicit boolean (it is True if the list is not empty)
            sum_eigvals += len(eigvals_dict[f'{key}'])
    
    assert sum_eigvals == N**2

def test_hamiltonianpart_eigvals(N):
    # Construct the Lindblad superoperator with the hamiltonian part only
    # N = 2
    RM_D = np.zeros((N**2 -1,N**2 -1))
    RM_H = tenpy.linalg.random_matrix.GUE((N,N))
    alpha, gamma = 1, 0

    # Compute the eigenvalues of the matrix that represents this superoperator in the HS base
    hamiltonianpart_eigvals = np.linalg.eigvals(Lindbladian_matrix(N,RM_D,RM_H,alpha,gamma))

    # Construct a dictionary that will contain the zero and notzero eigenvalues of the superoperator
    eigvals_dict = {'zero':[], 'notzero':[]}

    # Check if all the eigenvalues have null real part. For eigenvalues that also have null
    # imaginary part, append them in the dictionary at the 'zero' key, otherwhise append them
    # at the 'notzero' key
    for eigval in hamiltonianpart_eigvals:
        assert np.abs(np.real(eigval)) < 1e-10
        if np.abs(np.imag(eigval)) < 1e-10:
            eigvals_dict['zero'].append(eigval)
        else:
            eigvals_dict['notzero'].append(np.imag(eigval))

    # Check if the null eigenvalues are N
    assert len(eigvals_dict['zero']) == N

    # In order to check if the non zero eigenvalues are in complex and conjugate pairs, construct a
    # new list with these eigenvalues, rounded to 10 decimals and taken in absolute value, and from
    # it construct a set, so a list of unique elements. Finally check that each element in this set
    # appears two times in the initial list.
    abs_not_zero_eigvals = np.round(np.abs(eigvals_dict['notzero']),10)
    set_notzero_eigvals = set(abs_not_zero_eigvals)

    for item in set_notzero_eigvals:
        assert list(abs_not_zero_eigvals).count(item) == 2

def test_lindbladian_superoperator_eigvals(N):
    # N = 2
    RM_D = np.array(qutip.rand_dm_ginibre((N**2-1), rank=None))
    RM_H = tenpy.linalg.random_matrix.GUE((N,N))
    matrix = np.array(qutip.rand_dm_ginibre((N), rank=None))
    alpha, gamma = 1, 1

    # Compute the eigenvalues of the matrix that represents this superoperator in the HS base
    hamiltonianpart_eigvals = np.linalg.eigvals(Lindbladian(N,RM_D,RM_H,matrix,alpha,gamma))

    assert (sum(hamiltonianpart_eigvals) - 1) < 1e-9

def test_phit_eigvals(N):
    RM_D = np.array(qutip.rand_dm_ginibre((N**2-1), rank=None))
    RM_H = tenpy.linalg.random_matrix.GUE((N,N))
    alpha, gamma = 0.1, 0.1
    t = 1

    # Compute the matrix that represents the Lindblad superoperator in the HS base and construct
    # the associated channel phi(t)
    Lind_matr = Lindbladian_matrix(N,RM_D,RM_H,alpha,gamma)
    phit = phi_t(N,Lind_matr,t)

    eigvals_phit = np.linalg.eigvals(phit)
    
    eigvals_dict = {'one':[], 'real':[], 'cc_pairs':[]}

    for eigval in eigvals_phit:
        if (1 - np.abs(np.real(eigval))) < 1e-9 and np.abs(np.imag(eigval)) < 1e-10:
            eigvals_dict['one'].append(eigval)
        else:
            assert np.real(eigval) < 1.
            if np.abs(np.imag(eigval)) < 1e-10:
                eigvals_dict['real'].append(eigval)
            else:
                eigvals_dict['cc_pairs'].append(eigval)

    assert len(eigvals_dict['one']) == 1

    # Check if the sum of the eigenvalues is N**2
    sum_eigvals = 0
    for key in eigvals_dict.keys():
        if eigvals_dict[f'{key}']:  # implicit boolean (it is True if the list is not empty)
            sum_eigvals += len(eigvals_dict[f'{key}'])
    
    assert sum_eigvals == N**2

def test_choistate_eigvals(N):
    RM_D = np.array(qutip.rand_dm_ginibre((N**2-1), rank=None))
    RM_H = tenpy.linalg.random_matrix.GUE((N,N))
    alpha, gamma = 0.1, 0.1
    t = 1

    # Compute the matrix that represents the Lindblad superoperator in the HS base and construct
    # the associated channel phi(t)
    Lind_matr = Lindbladian_matrix(N,RM_D,RM_H,alpha,gamma)
    choi_state = choi_st(N,Lind_matr,t)

    eigvals_choistate = np.linalg.eigvals(choi_state)
    
    assert (sum(eigvals_choistate) - 1) < 1e-9


N = 3
for i in range(1000):
    print(i)
    test_dissipator_eigvals(N)
    test_hamiltonianpart_eigvals(N)
    test_lindbladian_superoperator_eigvals(N)
    test_phit_eigvals(N)
    test_choistate_eigvals(N)


