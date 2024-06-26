import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import normalize


""" Analysis.py function definitions """
def getHiCData_simulation(filepath):
    """
    Returns: 
        r: HiC Data
        D: Scaling
        err: error data 
    """
    contactMap = np.loadtxt(filepath)
    r=np.triu(contactMap, k=1) 
    r = normalize(r, axis=1, norm='max') 
    rd = np.transpose(r) 
    r=r+rd + np.diag(np.ones(len(r))) 

    D1=[]
    err = []
    for i in range(0,np.shape(r)[0]): 
        D1.append((np.mean(np.diag(r,k=i)))) 
        err.append((np.std(np.diag(r,k=i))))
    
    return(r,D,err)

def getHiCData_experiment(filepath, cutoff=0.0, norm="max"):
        """
        Returns: 
            r: HiC Data
            D: Scaling
            err: error data 
        """
        contactMap = np.loadtxt(filepath)
        r = np.triu(contactMap, k=1)
        r[np.isnan(r)]= 0.0
        r = normalize(r, axis=1, norm="max")
        
        if norm == "first":
            for i in range(len(r) - 1):
                maxElem = r[i][i + 1]
                if(maxElem != np.max(r[i])):
                    for j in range(len(r[i])):
                        if maxElem != 0.0:
                            r[i][j] = float(r[i][j] / maxElem)
                        else:
                            r[i][j] = 0.0 
                        if r[i][j] > 1.0:
                            r[i][j] = .5
        r[r<cutoff] = 0.0
        rd = np.transpose(r) 
        r=r+rd + np.diag(np.ones(len(r)))
    
        D1=[]
        err = []
        for i in range(0,np.shape(r)[0]): 
            D1.append((np.mean(np.diag(r,k=i)))) 
            err.append((np.std(np.diag(r,k=i))))
        D=np.array(D1)#/np.max(D1)
        err = np.array(err)
    
        return(r,D,err)

def greedy_descent(errors_file, initial_lr=0.02):
    R""" greedy descent towards a lower error value for the HiC map expiremental and HiC map simulation
    
    Args: 
        errors: (Numpy Array, required): 
            error data
        initial_lr: (float, optional):
            if theres not enough iterations in the error file it will use this defualt learning rate
    Returns: 
        lr: (float)
            next learning rate to use for the full inversion
    """
    lr = initial_lr
    lr_log = [lr]  # To track learning rate changes
    print(len(errors_file))
    for i in range(1, len(errors_file)):
        if len(errors_file) <= 5: # If theres less than 5 iterations use the default learning rate to generate more data
            lr_log.append(initial_lr)
        else:
            gradient = errors_file[i] - errors_file[i - 1]
            
            if gradient < 0:  # Error is decreasing
                lr *= 1.1  # Increase learning rate slightly
            else:  # Error is increasing or plateau
                lr *= 0.5  # Decrease learning rate by a bit punish for high errors
            
            # caps the learning rate between 0.001 - 0.1
            if len(errors_file <= 14):
                lr = max(min(lr, 0.1), 0.001)
            else:
                lr = max(min(lr, 0.5), 0.0005)
            lr_log.append(lr)

        return lr, lr_log
    
def refined_descent(errors, initial_lr=0.02):
        R""" more refined but slower descent towards a lower error value for the HiC map expiremental and HiC map simulation
    
        Args: 
            errors:
                np array of errors
        Returns: 
            lr: (float)
                next learning rate to use for the full inversion
        """
        lr = initial_lr
        lr_log = [lr]  # Track learning rate changes

        if len(errors) <= 5: # Use the default learning rate if not enough data
            lr_log.extend([initial_lr] * (len(errors) - 1))
        else:
            # Compute the mean gradient of all past errors
            gradients = np.diff(errors)
            mean_gradient = np.mean(gradients)

            # Adjust learning rate based on the mean gradient
            if mean_gradient < 0:  # Average error is decreasing
                lr *= 1.1  # Increase learning rate slightly
            else:  # Average error is increasing
                lr *= 0.6  # Significantly decrease learning rate

            # Cap the learning rate to ensure it remains within a certain range
            lr = max(min(lr, 0.1), 0.005)
            
            # Extend the log for all iterations with the adjusted rate
            lr_log.extend([lr] * (len(errors) - 1))

        return lr, lr_log
    


def get_error_data(error_file):
    """
    Extracts error data from a file. 
    
    Args:
        error_file (str): The path to the file containing error data.
    
    Returns:
        array: A array containing:
            - errors_log (np.array): Array of error data.
            - lr_log (np.array): Array of learning rate data.
            - mode (str): The mode of descent.
    """
    with open(error_file, 'r') as file:
        errors_log = []
        lr_log = []
        # mode_for_learningRate = []
        for line in file:
            parts = line.split()
            if len(parts) == 3:
                try:
                    errors_log.append(float(parts[0]))
                    lr_log.append(float(parts[1])) 
                    # mode_for_learningRate.append(str(parts[3]))
                except ValueError:
                    continue 
            elif len(parts) == 2:
                try:
                    errors_log.append(float(parts[0]))
                    lr_log.append(0.00)
                except ValueError:
                    continue
    # return np.array(errors_log), np.array(lr_log), mode_for_learningRate[-1]
    return np.array(errors_log), np.array(lr_log)
    
    
