import numpy as np
import pandas as pd
import os
import h5py

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


def ndb2cndb(filename):
    R"""
    Converts an **ndb** file format to **cndb**.
    
    Args:
        filename (path, required):
                Path to the ndb file to be converted to cndb.
    """
    Main_chrom      = ['ChrA','ChrB','ChrU'] # Type A B and Unknow
    Chrom_types     = ['ZA','OA','FB','SB','TB','LB','UN']
    Chrom_types_NDB = ['A1','A2','B1','B2','B3','B4','UN']
    Res_types_PDB   = ['ASP', 'GLU', 'ARG', 'LYS', 'HIS', 'HIS', 'GLY']
    Type_conversion = {'A1': 0,'A2' : 1,'B1' : 2,'B2' : 3,'B3' : 4,'B4' : 5,'UN' : 6}
    title_options = ['HEADER','OBSLTE','TITLE ','SPLT  ','CAVEAT','COMPND','SOURCE','KEYWDS','EXPDTA','NUMMDL','MDLTYP','AUTHOR','REVDAT','SPRSDE','JRNL  ','REMARK']
    model          = "MODEL     {0:4d}"
    atom           = "ATOM  {0:5d} {1:^4s}{2:1s}{3:3s} {4:1s}{5:4d}{6:1s}   {7:8.3f}{8:8.3f}{9:8.3f}{10:6.2f}{11:6.2f}          {12:>2s}{13:2s}"
    ter            = "TER   {0:5d}      {1:3s} {2:1s}{3:4d}{4:1s}"

    file_ndb = filename + str(".ndb")
    name     = filename + str(".cndb")

    cndbf = h5py.File(name, 'w')
    
    ndbfile = open(file_ndb, "r")
    
    loop = 0
    types = []
    types_bool = True
    loop_list = []
    x = []
    y = [] 
    z = []

    frame = 0

    for line in ndbfile:

        entry = line[0:6]

        info = line.split()


        if 'MODEL' in entry:
            frame += 1

            inModel = True

        elif 'CHROM' in entry:

            subtype = line[16:18]

            types.append(subtype)
            x.append(float(line[40:48]))
            y.append(float(line[49:57]))
            z.append(float(line[58:66]))

        elif 'ENDMDL' in entry:
            if types_bool:
                typelist = [Type_conversion[x] for x in types]
                cndbf['types'] = typelist
                types_bool = False

            positions = np.vstack([x,y,z]).T
            cndbf[str(frame)] = positions
            x = []
            y = []
            z = []

        elif 'LOOPS' in entry:
            loop_list.append([int(info[1]), int(info[2])])
            loop += 1
    
    if loop > 0:
        cndbf['loops'] = loop_list

    cndbf.close()
    return(name)

