""" In this program the most important functions of the library are tested.
"""
import numpy as np
import qutip
import tenpy
import unittest
from lindblad import *

class TestLindblad(unittest.TestCase):

    # N = 2

    def test_dissipator_eigvals(self,N):
        """ Test for the dissipator: tests if there are N**2 eigenvalues with the following properties:
            1) there always exist the null eigenvalue;
            2) all the eigenvalues have negative real part (except the null eigenvalue)
            3) the eigenvalues are in complex and conjugate pairs.
            Parameters: N : int
                            Dimension of the Hilbert space.
        """
        # Construct the Lindblad superoperator with the dissipator part only
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
                    self.assertEqual(np.sign(np.real(eigval)), -1,\
                                                "The eigenvalues should have negative real part")
                    eigvals_dict['real_negative'].append(eigval)

        # Check if there exist a single null eigenvalue
        self.assertEqual(len(eigvals_dict['zero']), 1, "There should always exist one null eigenvalue")

        # In order to check if the non zero eigenvalues are in complex and conjugate pairs, construct a
        # new list with these eigenvalues, rounded to 10 decimals and taken in absolute value, and from
        # it construct a set, so a list of unique elements. Finally check that each element in this set
        # appears two times in the initial list.
        abs_ccpairs_eigvals = np.round(np.abs(eigvals_dict['cc_pairs']),10)
        set_ccpairs_eigvals = set(abs_ccpairs_eigvals)

        if N != 2:
            for item in set_ccpairs_eigvals:
                self.assertEqual(list(abs_ccpairs_eigvals).count(item), 2, \
                                        "The eigenvalues should be in complex and conjugate pairs")
        
        # Check if the sum of the eigenvalues is N**2
        sum_eigvals = 0
        for key in eigvals_dict.keys():
            if eigvals_dict[f'{key}']:  # implicit boolean (it is True if the list is not empty)
                sum_eigvals += len(eigvals_dict[f'{key}'])
        
        self.assertEqual(sum_eigvals, N**2,\
                        f"The dimension of the Hilbert space is {N}; the eigenvalues must be {N**2}")
    
    def test_hamiltonianpart_eigvals(self,N):
        """ Test for the unitary hamiltonian contibution to the Lindbladian: tests if there are
            N**2 eigenvalues with the following properties:
            1) there always exist N null eigenvalues;
            2) all the eigenvalues null real part
            3) the eigenvalues are in complex and conjugate pairs.
            Parameters: N : int
                            Dimension of the Hilbert space.
        """
        # Construct the Lindblad superoperator with the hamiltonian part only
        RM_D = np.zeros((N**2 -1,N**2 -1))
        RM_H = tenpy.linalg.random_matrix.GUE((N,N))
        alpha, gamma = 1, 0

        # Compute the eigenvalues of the matrix that represents this superoperator in the HS base
        hamiltonianpart_eigvals = np.linalg.eigvals(Lindbladian_matrix(N,RM_D,RM_H,alpha,gamma))

        # Construct a dictionary that will contain the zero and nonzero eigenvalues of the superoperator
        eigvals_dict = {'zero':[], 'notzero':[]}

        # Check if all the eigenvalues have null real part. For eigenvalues that also have null
        # imaginary part, append them in the dictionary at the 'zero' key, otherwhise append them
        # at the 'notzero' key
        for eigval in hamiltonianpart_eigvals:
            self.assertLess(np.abs(np.real(eigval)), 1e-10,\
                                                "All the eigenvalues should have null real part")
            if np.abs(np.imag(eigval)) < 1e-10:
                eigvals_dict['zero'].append(eigval)
            else:
                eigvals_dict['notzero'].append(np.imag(eigval))

        # Check if the null eigenvalues are N
        self.assertEqual(len(eigvals_dict['zero']), N, f"There should be {N} null eigenvalues")

        # In order to check if the non zero eigenvalues are in complex and conjugate pairs, construct a
        # new list with these eigenvalues, rounded to 10 decimals and taken in absolute value, and from
        # it construct a set, so a list of unique elements. Finally check that each element in this set
        # appears two times in the initial list.
        abs_not_zero_eigvals = np.round(np.abs(eigvals_dict['notzero']),10)
        set_notzero_eigvals = set(abs_not_zero_eigvals)

        for item in set_notzero_eigvals:
            self.assertEqual(list(abs_not_zero_eigvals).count(item), 2,\
                                    "The eigenvalues should be in complex and conjugate pairs")

    def test_lindbladian_superoperator_eigvals(self,N):
        """ Test for the Lindbladian superoperator: tests if such superoperator acting on a state
            gives as output a well-defined physical state. In order to do it it is verified if the
            eigenvalues of the output matrix are positive and if they sum up to one.
        """
        RM_D = np.array(qutip.rand_dm_ginibre((N**2-1), rank=None))
        RM_H = tenpy.linalg.random_matrix.GUE((N,N))
        matrix = np.array(qutip.rand_dm_ginibre((N), rank=None))
        alpha, gamma = 1, 1

        # Compute the eigenvalues of the output matrix of the Lindbladian
        lindbladian_output_eigvals = np.linalg.eigvals(Lindbladian(N,RM_D,RM_H,matrix,alpha,gamma))

        # Check if these eigenvalues are positive and sum up to one
        for eigval in lindbladian_output_eigvals:
            self.assertGreater(eigval, 0., "The eigenvalues of a state must be positive")
        self.assertLess( (sum(lindbladian_output_eigvals) - 1), 1e-9,\
                                           "The eigenvalues of a state must sum up to one")


if __name__ == '__main__':
    N = 3
    unittest.main()
