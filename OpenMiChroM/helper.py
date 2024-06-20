import numpy as np
import pandas as pd
import os

""" Optimization.py function definitions """

def normalize_matrix(matrix):
    R"""
    Normalize the matrix for simulation optimization. Here the first neighbor should have the probability of contact P=1.0.
    """
    matrix = np.nan_to_num(matrix, nan=0, posinf=0, neginf=0)
    np.fill_diagonal(matrix,0.0)

    max_values = np.amax(np.triu(matrix), axis=1)
    
    # To avoid division by zero, replace zeros with ones
    max_values[max_values == 0] = 0.0000001
    
    normalized_matrix = np.triu(matrix) / max_values[:, np.newaxis]
    # return normalized_matrix
    matrix= normalized_matrix + np.triu(normalized_matrix,k=1).T
    np.fill_diagonal(matrix,1.0)

    return matrix


def knight_ruiz_balance(matrix, tol=1e-5, max_iter=100):
    R"""
    Perform the Knight-Ruiz matrix balancing.
    """
    A = np.array(matrix, dtype=float)
    n = A.shape[0]
    row_scaling = np.ones(n)
    col_scaling = np.ones(n)
    for _ in range(max_iter):
        row_scaling = np.sqrt(np.sum(A, axis=1))
        A /= row_scaling[:, None]
        col_scaling = np.sqrt(np.sum(A, axis=0))
        A /= col_scaling

        if np.all(np.abs(row_scaling - 1) < tol) and np.all(np.abs(col_scaling - 1) < tol):
            break

    return A

    

def saveLambdas(sequenceFile, data, outputPath, name):
    seq = np.loadtxt(sequenceFile, dtype=str)[:,1]

    lamb = pd.DataFrame(data,columns=seq)
    lamb.to_csv(os.path.join(outputPath, name), index=False)
    print("{} file save in {}".format(name, outputPath))
    
def get_chrom_seq(filename):
    R"""Reads the chromatin sequence as letters of the types/compartments.
    
    Args:

        filename (file, required):
            Chromatin sequence of types file. The first column should contain the locus index. The second column should have the locus type annotation. A template of the chromatin sequence of types file can be found at the `Nucleome Data Bank (NDB) <https://ndb.rice.edu/static/text/chr10_beads.txt>`_.
            
    Returns:
        :math:`(N,1)` :class:`numpy.ndarray`:
            Returns an array of the sequence of chromatin types.

    """
    my_list = []
    af = open(filename,'r')
    pos = af.read().splitlines()
    for t in range(len(pos)):
        pos[t] = pos[t].split()
        my_list.append(pos[t][1])

    return np.array(my_list)


""" CndbTools.py function definitions """


    