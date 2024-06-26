import numpy as np
import os
from OpenMiChroM.CndbTools import cndbTools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import util

file = "Development"

class Ana:
    def __init__(self):
        """
        Initializes the Ana class with a base folder for data storage.
        """
        self.datasets = {}
        self.cndbTools = cndbTools()
        self.outPath = os.path.join(os.getcwd(), 'Analysis')
        os.makedirs(f'{os.path.join(os.getcwd(), 'Analysis')}', exist_ok=True)

    if file != "Development":
        def FullInversion_lr(self, filename):
            """
            Reads error data from a file and prints the error array, learning rate array, and mode.
            
            Args:
                filename (str): The path to the file containing error data.
            """
            error_array, lr_array, mode = util.get_error_data(filename)
            print(error_array, lr_array, mode)

    def add_dataset(self, label, folder):
        """
        Adds a dataset to the analysis.
        
        Args:
            label (str): The label for the dataset.
            folder (str): The folder path containing the dataset.
            
        """
        self.datasets[label] = {
            'folder': str(folder),
            'trajectories': None, # Trajectory data for the dataset
            'distance_array': None, # euclidian distance array
        }
        
    def HiCMapPlot(self, hic_exp, hic_sim, outputFileName='HiC_exp_vs_sim.png', show=True, figsize=10):
        """
        Plots a Hi-C map comparison between simulated and experimental data.
        
        Args:
            hic_exp (array or filepath): Experimental Hi-C data. expects an array gotten from util.getHiCData_expirement or a filepath to the chromosome .dense file
            hic_sim (array or filepath): Simulation Hi-C data expects an array gotten returned from util.getHiCData_simulation or a filepath to the simulations probdist file
            outputFileName (filepath, optional): File name to save as defaults to HiC_exp_vs_sim.png.
            show (bool, optional): Whether to show the plot. Default is True.
            figsize (int, optional): Size of the figure. Default is 10.
        """
        # check if a string (filename) is passed if so retrieve the hic data from the function else use the array passed onto HiCMapPlot
        if isinstance(hic_sim, str):
            hic_sim, _, __ = util.getHiCData_simulation(hic_sim)
        if isinstance(hic_exp, str):
            hic_exp, _, __ = util.getHiCData_experiment(hic_exp, norm="first")
        
        comp = np.triu(hic_exp) + np.tril(hic_sim, k=1)
        plt.rcParams["figure.figsize"] = (figsize, figsize)
        mpl.rcParams['axes.linewidth'] = 2

        plt.matshow(comp, norm=mpl.colors.LogNorm(vmin=0.001, vmax=1.0), cmap='Reds')
        plt.title(f"Simulated vs. Experimental Hi-C", pad=20)  
        plt.colorbar()  
        plt.savefig(f'{self.outPath}/{outputFileName}')
        if show:
            plt.show()
    
    def GenomeDistancePlot(self, scale_sim, scale_exp, outputFileName='genomicDistance.png', show=True):
        """
        Plots the contact probability versus genomic distance.
        
        Args:
            scale_exp (array or filepath): Experimental scaling data.
            scale_sim (array or filepath): Experimental scaling data.
            outputFileName (str): Filename to save results as.
           
            show (bool, optional): Whether to show the plot. Default is True.
        """
        if isinstance(scale_sim, str):
            _, scale_sim, __ = util.getHiCData_simulation(scale_sim)
        if isinstance(scale_exp, str):
            _, scale_exp, __ = util.getHiCData_experiment(scale_exp, norm="first")
        
        mpl.rcParams['axes.linewidth'] = 2.
        cmap = sns.blend_palette(['white', 'red'], as_cmap=True)  
        cmap.set_bad(color='white')
        fig, ax = plt.subplots(figsize=(10, 10))  

        ax.loglog(range(len(scale_exp)), scale_exp, color='r', label='Exp.', linewidth=2)
        ax.loglog(range(len(scale_sim)), scale_sim, color='green', linestyle='--', label='Simulated', linewidth=2)
        ax.set_title('Scaling', loc='center')
        ax.set_xlabel('Genomic Distance')
        ax.set_ylabel('Probability')
        ax.set_xlim([1, len(scale_exp)])  
        ax.set_ylim([min(scale_exp), max(scale_exp)])  
        ax.legend()
        
        plt.savefig(f'{self.outPath}/{outputFileName}')
        if show:
            plt.show()
    
    def make_error_plot(self, error_file, outputFileName='error.png', show=True):
        """
        Plots an error graph for the optimization simulations 
        expects an error file in the format of {iteration} {error}
                                               {iteration} {error}
        outputFileName: (optional)
            Filename to save figure as.
        show: (optional, boolean):
            Whether to show the figure or not default is True.
        Args:
            error_file (str): The path to the file containing error data.
            show (boolean)
        """
        error_data = np.loadtxt(error_file, dtype=float)
        error_data = error_data[:, 1]  # Return the second column, which contains the error
        
        iterations = np.arange(max(1, len(error_data)))
        # iterations_eta = np.arange(max(1, len(eta_log)))

        # Plot error graph
        plt.subplot(1, 2, 1)
        plt.plot(iterations, error_data, marker='o', linestyle='-', color='b', label='Error')
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title('Error over Iterations')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{self.outPath}/{outputFileName}')
        
        if show:
            plt.show()
        return
   

    def process_trajectories(self, label, filename, folder_pattern=['iteration_', [0,100]]):
        """
        Processes trajectory data for a given dataset.
        
        Args:
            label (str, required): The label for the dataset.
            filename (str, required): The filename of the trajectory data (.cndb file).
            num_replicas (int): the number of iterations done on the simulations makes it easier if the directory holding your simulation data is
            folder/{}_0, folder/{}_1 .. folder/{}_{num_replicas}
            file_pattern (array: [filepattern name, [iteration start, iteration end]], required)

            
            * If 0 is given as the paramater for num_replicas then it will try and load a single .cndb trajectory file
        """
        config = self.datasets[label]
        trajs_xyz = []
        it_start = folder_pattern[1][0]
        it_end = folder_pattern[1][1]

                    
        inputFolder = os.path.join(config['folder'], folder_pattern[0])
        
        
        if it_start == it_end:                                    #### RIGHT HERE
            traj = self.__load_and_process_trajectory(folder=inputFolder, replica=it_start, filename=filename)
            if traj.size > 0:
                self.datasets[label]['trajectories'] = np.vstack(traj)
                print(f'Trajectory for {label} has shape {self.trajectories[label].shape}')
            else:
                print(f"No valid trajectories found for {label} at the file {inputFolder}{it_start}/{filename}")

            
        for i in range(it_start, it_end + 1):
            traj = self.__load_and_process_trajectory(
                folder=inputFolder, replica=i, filename=filename
            )
            if traj.size > 0:
                trajs_xyz.append(traj)
        
        if trajs_xyz:
            self.datasets[label]['trajectories'] = np.vstack(trajs_xyz) # Store the traj_xyz data in the label given
            print(f'Trajectory for {label} has shape {self.datasets[label]["trajectories"].shape}')
        else:
            print(f"No valid trajectories found for {label}")

    def __load_and_process_trajectory(self, folder, replica, filename, key=None):
        """
        Loads and processes a single trajectory file.
        
        Args:
            folder (str): The folder containing the trajectory file.
            replica (int): The replica number.
            filename (str): The filename of the trajectory data.
            key (str, optional): Key for bead selection.
        
        Returns:
            np.array: Processed trajectory data.
        """
        path = f'{folder}{replica}/{filename}'
        
        if not os.path.exists(path):
            print(f"File does not exist: {path}")
            return np.array([])
        else:
            print(f"Processing file: {path}")

        try:
            trajectory = self.cndbTools.load(filename=path)            
            list_traj = [int(k) for k in trajectory.cndb.keys() if k != 'types']
            list_traj.sort()
            beadSelection = trajectory.dictChromSeq[key] if key else None
            first_snapshot, last_snapshot = list_traj[0], list_traj[-1]
            trajs_xyz = self.cndbTools.xyz(frames=[first_snapshot, last_snapshot+1, 2000], XYZ=[0,1,2], beadSelection=beadSelection)
            return trajs_xyz
        
        except Exception as e:
            print(f"Error processing trajectory {replica}: {str(e)}")
            return np.array([])
        
    """====================================================================== CLUSTERING ===================================================================================="""

    def create_dendogram(self, *args, outputFileName='dendogram.png', show=True, _clusterOnly=False, k_offset=500, metric='euclidean'):
        """
        Args:
            show (boolean, required): whether to show the dendogram
            outputFileName (filepath, optional): name to save plot as
            k_offset (int, required): the offset parameter for flattening the distance matrices. Defaults to 500.
            metric (str, required): the distance metric to use for computing pairwise distances. Defaults to 'euclidean'.
            _clusterOnly (boolean, dont use): used by cluster function to get the cluster threshold
            **args: (strings, required): the labels to create the dendogram from
            
        Usage Example: create_dendogram('
                                    outputFileName=dendogram.png', show=True, k_offset=500, method='euclidean'
        **args the labels ->        'compartment', 'subcompartment', 'homopolymer')
        """
        
        if len(args) == 0:
            print("No arguments given")
            return
        
        flat_euclid_dist_map = {}
        
        for label in args:
            print(f'processing {label}')
            trajectories = self.datasets[label]['trajectories']
            if trajectories is None or len(trajectories) == 0:
                print(f"Trajectories not yet loaded for {label}. Load them first")
                return
            # Compute pairwise Euclidean distances
            dist = [cdist(val, val, 'euclidean') for val in trajectories]
            dist = np.array(dist)
            print(f"{label} has dist shape {dist.shape}")
            
            # Store in datasets label 
            self.datasets[label]["distance_array"] = dist
            flat_euclid_dist_map[label] = dist
        
        # Flatten the distance arrays
        flat_euclid_dist_map = {label: [flat_euclid_dist_map[label][val][np.triu_indices_from(flat_euclid_dist_map[label][val], k=k_offset)].flatten()
                                        for val in range(len(flat_euclid_dist_map[label]))]
                                for label in args}
        
        # Make it into a 1D vertical array
        X = np.vstack([item for sublist in flat_euclid_dist_map.values() for item in sublist])
        print(f"Flattened distance array has shape: {X.shape}")
        
        # Create the linkage matrix
        Z = linkage(X, method="weighted", metric=metric)
        
        if _clusterOnly:
            return X, Z
        
        plt.figure(figsize=(10, 7))
        dn = dendrogram(Z)
        
        # Plot the dendogram
        plt.title("Dendrogram")
        plt.savefig(os.path.join(self.outPath, outputFileName))
        if show:
            plt.show()

        
    def create_euclidian_dist_map(self, label, show=True, outputFileName="euclid_dist.png", cmap='viridis_r'):
        """
        Args:
            label (string): label of the dataset to create the euclidian_dist_plot
            show (boolean, optional): whether to show the dendogram
            outputFileName (filepath, optional): name to save plot as'
        """
        # if not already defined compute the distance
        trajectories = self.datasets[label]['trajectories']

        if len(trajectories) == 0:
                print(f"Trajectories not yet loaded for {label}. Load them first")
                return
            
        if len(self.datasets[label]["distance_array"])== 0:
            compute_dist = [cdist(val, val, 'euclidean') for val in trajectories]
            compute_dist = np.array(compute_dist)
            self.datasets[label]['distance_array'] = compute_dist
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        p = ax.imshow(self.datasets[label]['distance_array'][0], vmin=0, vmax=25, cmap=cmap)
        plt.colorbar(p, ax=ax, fraction=0.046)
        plt.title(f'Euclidean Distance Map {label}')
        plt.xlabel('Point Index')
        plt.ylabel('Point Index')
        
        # Save figure
        plt.savefig(os.path.join(self.outPath, outputFileName))
        
        # Show the figure if True
        if show:
            plt.show()
        
    def PCA_plot(self, *args, outputFileName="PCA", outputPlotName='dendogram.png', show=True):
        """
        Args:
            show (boolean, optional): Whether to show the cluster plot. Defaults to True.
            outputPlotName (str, optional): Name to save the plot as. Defaults to 'dendogram.png'.
            outputFileName (str, required): Name to save data as.
            **args: (strings, required): The labels to create the cluster from.
            
        Usage Example:
            PCA_plot('compartment', 'subcompartment', 'homopolymer', outputFileName='output.csv', show=True)
        """
        num_clusters = len(args)
        flattened_distance_array, linkage_matrix = self.create_dendogram(show=False, _clusterOnly=True, *args)
        threshold = linkage_matrix[-num_clusters, 2]
        print(f"Threshold for {num_clusters} clusters: {threshold}")
        fclust = fcluster(linkage_matrix, t=threshold, criterion='distance')
        print(f"Cluster assignments: {fclust}")
        np.savetxt(os.path.join(self.outPath, f"fclust_{outputFileName}.txt"), fclust)
        
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(flattened_distance_array)
        principalDF = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
        principalDF.to_csv(os.path.join(self.outPath, f'balance_pca_{outputFileName}.csv'))
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        
        # Plot the PCA results
        cmap = 'viridis'
        marker_size = 2
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        scatter = ax.scatter(principalDF["PC1"], principalDF["PC2"], c=fclust, alpha=0.5, cmap=cmap, s=marker_size)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ticks = np.arange(1, num_clusters + 1)
        cbar = plt.colorbar(scatter)
        cbar.set_ticks(ticks)
        plt.savefig(os.path.join(self.outPath, f'PCAs_{outputPlotName}.pdf'))
        if show:
            plt.show()
        plt.close()

        sizes = {label: self.datasets[label]['distance_array'].shape[0] for label in args}
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        start = 0
        colors = ['tab:orange', 'tab:green', 'tab:red']
        for idx, label in enumerate(args):
            end = start + sizes[label]
            scatter = ax.scatter(principalDF["PC1"][start:end], principalDF["PC2"][start:end], alpha=0.5, c=colors[idx], label=label, s=marker_size)
            start = end
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.legend(bbox_to_anchor=(0., 1.01, 1., .1), 
                loc='lower left',
                ncol=1, 
                borderaxespad=0.,
                frameon=False)
        plt.savefig(os.path.join(self.outPath, f'PCAs_{outputPlotName}_per_ensemble.pdf'))
        if show:
            plt.show()
        plt.close()