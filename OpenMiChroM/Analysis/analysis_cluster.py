#
# Run cluster analysis
#
from OpenMiChroM.Optimization import CustomMiChroMTraining 
from OpenMiChroM.CndbTools import cndbTools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
from sklearn.preprocessing import normalize
import math
import numpy as np
import pandas as pd
from pandas import read_csv
import h5py
# import hdf5plugin  
import sys
from scipy import sparse
from pathlib import Path
import glob
from os import path
import os
import scipy as sc
import argparse
import random
import fileinput
import itertools
import linecache
from itertools import islice
from scipy.spatial import distance 
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from scipy.signal import savgol_filter
from scipy import stats
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
from multiprocessing import Pool
import multiprocessing
from functools import partial
from numpy import linalg as LA
from scipy.stats import entropy
from scipy.cluster.hierarchy import fclusterdata, fcluster
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE

kBinv=1/sc.constants.R*1000

cndbTools = cndbTools()

# for trajectory selection
snapshot = 1000  #1000  # 100
# for t-SNE
perplexity = 500  # 660
n_iter = 5000
k_offset = 500 # 50 # 100, 500, 1000, 2000

save = "/work/cms16/ro21/optimization/balance/" + str(snapshot) + "_" + str(perplexity) + "_k_" + str(k_offset) + "/"
outname = 'type_10_Tic_production'
num_clusters = 3

print(save)

# Use Path.mkdir to create the directory
path = Path(save)
path.mkdir(parents=True, exist_ok=True)

num_processes = multiprocessing.cpu_count()
print('Number of processors available is ', num_processes)

def calcQ(r1,r2):
    sigma=2*np.ones(shape=comp_dist[0].shape)
    return np.exp(-(r1-r2)**2/(sigma[np.triu_indices_from(sigma)].flatten())**2).mean()

def process_trajectory(i, key=None):
    trajectory = cndbTools.load(folder + '/output_' + str(iteration) 
                                + '_' + str(i) + '/opt_0.cndb')

    list_traj = [int(key) for key in trajectory.cndb.keys() 
                    if not key == 'types']
    list_traj.sort()
    first_snapshot = list_traj[0]
    last_snapshot  = list_traj[-1]

    if not key == None:
        beadSelection = trajectory.dictChromSeq[key]
    else:
        beadSelection = None

    trajs_xyz = cndbTools.xyz(frames=[first_snapshot, last_snapshot+1, snapshot], 
                              beadSelection=beadSelection,
                              XYZ=[0,1,2])

    return trajs_xyz


# ic_10+types

# base folder with the replicas
folder = "/work/cms16/ro21/optimization/AB_sub_50k/chr10/ic_conf/lambda_0_2712/1e-05/types_10/3e-07/"
# iteration
iteration = 67
# total number of replicas
number_of_replicas = 32
# analysis output folder
output = folder + "/analysis/"

print("Reading trajectories...")
# Read trajectories of iteration with Number_of_replicas
process_trajectory_partial = partial(process_trajectory, key=None)
with multiprocessing.Pool(num_processes) as p:
    trajs_xyz = p.map(process_trajectory_partial, range(1, number_of_replicas+1))
all_rep_compartment = np.vstack(trajs_xyz)
print('Trajectory/all_rep shape is ', all_rep_compartment.shape)


# types+icC

# base folder with the replicas
folder = "/work/cms16/ro21/optimization/AB_sub_50k/chr10/original/"
# iteration
iteration = 0
# total number of replicas
number_of_replicas = 32
# analysis output folder
output = folder + "/analysis/"

print("Reading trajectories...")
# Read trajectories of iteration with Number_of_replicas
process_trajectory_partial = partial(process_trajectory, key=None)
with multiprocessing.Pool(num_processes) as p:
    trajs_xyz = p.map(process_trajectory_partial, range(1, number_of_replicas+1))
all_rep_subcompartment = np.vstack(trajs_xyz)
print('Trajectory/all_rep shape is ', all_rep_subcompartment.shape)


# homopolymer

# base folder with the replicas
folder = "/work/cms16/ro21/gm12878/homopolymer_Types/A1/chr10"
# iteration
iteration = 0
# total number of replicas
number_of_replicas = 24
# analysis output folder
output = folder + "/analysis/"

snapshot = int(snapshot*(3/4))

print("Reading trajectories... snapshots=", snapshot)
# Read trajectories of iteration with Number_of_replicas
process_trajectory_partial = partial(process_trajectory, key=None)
with multiprocessing.Pool(num_processes) as p:
    trajs_xyz = p.map(process_trajectory_partial, range(1, number_of_replicas+1))
all_rep_homopolymer = np.vstack(trajs_xyz)
print('Trajectory/all_rep shape is ', all_rep_homopolymer.shape)


# Using list comprehension to compute pairwise Euclidean distances
comp_dist = [distance.cdist(val, val, 'euclidean') for val in all_rep_compartment]
comp_dist = np.array(comp_dist)
print("comp_dist has shape ", comp_dist.shape)

subcomp_dist = [distance.cdist(val, val, 'euclidean') for val in all_rep_subcompartment]
subcomp_dist = np.array(subcomp_dist)
print("subcomp_dist has shape ", subcomp_dist.shape)

# Using list comprehension to compute pairwise Euclidean distances
homo_dist = [distance.cdist(val, val, 'euclidean') for val in all_rep_homopolymer]
size = comp_dist.shape[0]
homo_dist = np.array(homo_dist[:size])
print("homo_dist has shape ", homo_dist.shape)
homo_dist.shape

# Create a flatten distance array for each frame
X1 = [comp_dist[val][np.triu_indices_from(comp_dist[val],k=k_offset)].flatten() 
      for val in range(len(comp_dist))]
X2 = [subcomp_dist[val][np.triu_indices_from(subcomp_dist[val],k=k_offset)].flatten()
      for val in range(len(comp_dist))]
X3 = [homo_dist[val][np.triu_indices_from(homo_dist[val],k=k_offset)].flatten() 
      for val in range(len(homo_dist))]
X = np.vstack((X1,X2,X3))
print("Flatten distance array has shape ", X.shape)

# Create the dendogram
print("Creating the dendogram...")
# # Z = linkage(X, method="weighted", metric=calcQ) 
Z = linkage(X, method="weighted", metric='euclidean')
file_Z = save + '/dendogram_' + outname
np.savetxt(file_Z, Z)

# plot Z
fig = plt.figure(figsize=(10, 4))
dn = dendrogram(Z)
plt.savefig(save + '/dendogram_' + outname + '.pdf')
plt.close()


# Find the threshold for the clusters
threshold = Z[-(num_clusters), 2]
print("Threshold for", num_clusters, "clusters:", threshold)

# Assign clusters using fcluster
fclust = fcluster(Z, t=threshold, criterion='distance')
print("Cluster assignments:", fclust)
file_fclust = save + '/fclust_' + outname
np.savetxt(file_fclust, fclust)

# Create the PCAs
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data=principalComponents, columns = ['PC1', 'PC2'])
principalDf.to_csv(save + "/balance_pca_" + outname + ".csv", index=False)

print(pca.explained_variance_ratio_)

# plot()
cmap = 'viridis'
marker_size = 2
fig,ax = plt.subplots(1,1, figsize=(4,3))
scatter = ax.scatter(principalDf["PC1"], principalDf["PC2"], c=fclust, alpha=0.5, cmap=cmap, s=marker_size)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ticks=np.arange(1,num_clusters+1)
cbar = plt.colorbar(scatter)
cbar.set_ticks(ticks)
plt.savefig(save + '/PCAs_' + outname + '.pdf')
plt.close()

s1 = comp_dist.shape[0]
s2 = subcomp_dist.shape[0]
s3 = homo_dist.shape[0]
fig,ax = plt.subplots(1,1, figsize=(4,4))
scatter = ax.scatter(principalDf["PC1"][:s1], principalDf["PC2"][:s1], alpha=0.5, c="tab:orange", label='ic$_{10}$ + types', s=marker_size)
scatter = ax.scatter(principalDf["PC1"][s1+1:s1+s2], principalDf["PC2"][s1+1:s1+s2], alpha=0.5, c="tab:green", label='types + ic', s=marker_size)
scatter = ax.scatter(principalDf["PC1"][s1+s2+1:], principalDf["PC2"][s1+s2+1:], alpha=0.5, c="tab:red", label='homopolymer', s=marker_size)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.legend(bbox_to_anchor=(0., 1.01, 1., .1), 
                            loc='lower left',
                            ncol=1, 
                            borderaxespad=0.,
                            frameon=False)
plt.savefig(save + '/PCAs_' + outname + '_per_ensemble.pdf')
plt.close()


# t-SNE
tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, metric='euclidean', n_iter=n_iter)
tsne_results = tsne.fit_transform(X)
tsneDf = pd.DataFrame(data=tsne_results, columns = ['t-SNE1', 't-SNE2'])
tsneDf.to_csv(save + "/balance_tsne_" + outname + ".csv", index=False)

# plot()
fig,ax = plt.subplots(1,1, figsize=(4,3))
scatter = ax.scatter(tsne_results[:,0], tsne_results[:,1],c=fclust,alpha=0.5,cmap='viridis', s=marker_size)
ax.set_xlabel("t-SNE1")
ax.set_ylabel("t-SNE2")
ticks=np.arange(1,num_clusters+1)
cbar = plt.colorbar(scatter)
cbar.set_ticks(ticks)
fig.savefig(save + '/tSNE_' + outname + '.pdf')
plt.close()

fig,ax = plt.subplots(1,1, figsize=(4,4))
ax.scatter(tsne_results[:s1,0], tsne_results[:s1,1],alpha=0.5,c="tab:orange", label='ic$_{10}$ + types', s=marker_size)
ax.scatter(tsne_results[s1+1:s1+s2,0], tsne_results[s1+1:s1+s2,1],alpha=0.5,c="tab:green", label='types + ic', s=marker_size)
ax.scatter(tsne_results[s1+s2+1:,0], tsne_results[s1+s2+1:,1],alpha=0.5,c="tab:red", label='homopolymer', s=marker_size)
ax.set_xlabel("t-SNE1")
ax.set_ylabel("t-SNE2")
plt.legend(bbox_to_anchor=(0., 1.01, 1., .1), 
                            loc='lower left',
                            ncol=1,
                            borderaxespad=0., 
                            frameon=False)
fig.savefig(save + '/tSNE_' + outname + 'per_ensemble.pdf')
plt.close()

