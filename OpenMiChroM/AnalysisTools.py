import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.utils import resample
import seaborn as sns

from OpenMiChroM.CndbTools import cndbTools


class Ana:
    def __init__(self, outputFolderPath=None):
        """
        Initializes the Ana class with a base folder for data storage.
        """
        # To not overcompute the linkage_matrix, flattened_distance_array of a set of objects if it already has them 
        self.cache = {} # holds a sorted tuple and the linkage (Z), dist array (X)
        self.datasets = {}
        self.cndbTools = cndbTools()
        
        if outputFolderPath == None:
            self.outPath = os.path.join(os.getcwd(), 'Analysis')
            os.makedirs(f'{os.path.join(os.getcwd(), 'Analysis')}', exist_ok=True) # make analysis folder to store plots data etc.
        else:
            self.outPath = os.path.join(os.getcwd(), outputFolderPath)
            os.makedirs(f'{os.path.join(os.getcwd(), outputFolderPath)}', exist_ok=True) # make folder at custom path to store plots data etc.

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
            hic_exp (array or filepath): Experimental Hi-C data. expects an array gotten from self.getHiCData_expirement or a filepath to the chromosome .dense file
            hic_sim (array or filepath): Simulation Hi-C data expects an array gotten returned from self.getHiCData_simulation or a filepath to the simulations probdist file
            outputFileName (filepath, optional): File name to save as defaults to HiC_exp_vs_sim.png.
            show (bool, optional): Whether to show the plot. Default is True.
            figsize (int, optional): Size of the figure. Default is 10.
        """
        # check if a string (filename) is passed if so retrieve the hic data from the function else use the array passed onto HiCMapPlot
        if isinstance(hic_sim, str):
            hic_sim, _, __ = self.getHiCData_simulation(hic_sim)
        if isinstance(hic_exp, str):
            hic_exp, _, __ = self.getHiCData_experiment(hic_exp, norm="first")
        
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
            _, scale_sim, __ = self.getHiCData_simulation(scale_sim)
        if isinstance(scale_exp, str):
            _, scale_exp, __ = self.getHiCData_experiment(scale_exp, norm="first")
        
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

    def create_dendogram(self, *args, outputPlotName='dendogram.png', show=True, metric='euclidean', method='weighted', plot_params=None):
        """
        Args:
            show (boolean, required): whether to show the dendogram
            outputPlotName (filepath, optional): name to save plot as
            metric (str/function, required): the distance metric to use for computing pairwise distances. Defaults to 'euclidean'.
            method (str, required): Defaults to weighted
            plot (dict, optional): overwrite plot parameters such as title, figsize, label, etc.
            *args: (strings, required): the labels to create the dendogram for
            
        Usage Example: create_dendrogram(
                                    outputPlotName='dendrogram.png', show=True, k_offset=500, metric='euclidean', method='weighted',
                                    plot={'figsize': (10, 4), 'title': 'My Dendrogram', 'label': 'dka'},
                                    'compartment', 'subcompartment', 'homopolymer')
        """
        
        if len(args) == 0:
            print("No arguments given")
            return
        
        X, Z = self.calc_XZ(*args, methodd=method, metricc=metric)
        del X
        
        default_plot_params = {
            'figsize': (10, 7),
            'title': 'Dendogram'
        }
        
        if plot_params is not None:
            default_plot_params.update(plot_params)
            
        
        plt.figure(figsize=default_plot_params['figsize'])
        dn = dendrogram(Z)
        
        # Plot the dendogram
        plt.title(default_plot_params['title'])
        plt.savefig(os.path.join(self.outPath, outputPlotName))
        if show:
            plt.show()

        
    def create_euclidian_dist_map(self, label, show=True, outputFileName="euclid_dist.png", cmap='viridis_r', plot_params=None):
        """
        Args:
            label (string): label of the dataset to create the euclidian_dist_plot
            show (boolean, optional): whether to show the dendogram
            outputFileName (filepath, optional): name to save plot as'
            plot_params (dict, optional): overwrite plot parameters such as title, figsize, label, etc.

        """
        # if not already defined compute the distance
        trajectories = self.datasets[label]['trajectories']

        if trajectories.all() == None:
                print(f"Trajectories not yet loaded for {label}. Load them first")
                return
            
        if len(self.datasets[label]["distance_array"]) == 0:
            compute_dist = [cdist(val, val, 'euclidean') for val in trajectories]
            compute_dist = np.array(compute_dist)
            self.datasets[label]['distance_array'] = compute_dist
            
        def_plt_params = {
            'figsize':(10,8),
            'cmap': 'viridis_r',
            'vmin': 0,
            'vmax': 25,
            'fraction':0.046,
            'xlabel':'Point Index',
            'ylabel': 'Point Index',
            'title': f'Euclidian Distance Map {label}'
        }
        
        if plot_params != None:
            def_plt_params.update(plot_params)

        
        fig, ax = plt.subplots(1, 1, figsize=def_plt_params['figsize'])
        p = ax.imshow(self.datasets[label]['distance_array'][0], vmin=def_plt_params['vmin'], vmax=def_plt_params['vmax'], cmap=def_plt_params['cmap'])
        plt.colorbar(p, ax=ax, fraction=def_plt_params['fraction'])
        plt.title(def_plt_params['title'])
        plt.xlabel(def_plt_params['xlabel'])
        plt.ylabel(def_plt_params['ylabel'])
        
        # Save figure
        plt.savefig(os.path.join(self.outPath, outputFileName))
        
        # Show the figure if True
        if show:
            plt.show()
        
    def PCA_plot(self, *args, outputFileName="PCA", outputPlotName='dendogram.png', show=True, combined_pca=None, separated_pca=None, method='weighted', metric='euclidean'):
        """
        Args:
            show (boolean, optional): Whether to show the cluster plot. Defaults to True.
            outputPlotName (str, optional): Name to save the plot as. Defaults to 'dendogram.png'.
            outputFileName (str, required): Name to save data as.
            combined_pca (dict, optional): Parameters for the combined PCA plot.
            separated_pca (dict, optional): Parameters for the separated PCA plot.
            *args: (strings, required): The labels to create the cluster from.

        Usage Example:
            combined_pca_plot_params = {'figsize': (8, 8), 'cmap': 'plasma', 'marker_size': 50, 'title': 'Custom Combined PCA Plot'}
            separated_pca_plot_params = {'figsize': (8, 8), 'colors': ['tab:blue', 'tab:orange', 'tab:green'], 'marker_size': 50, 'title': 'Custom Separated PCA Plot'} 
             
            PCA_plot('compartment', 'subcompartment', 'homopolymer', outputFileName='output', show=True,
                    combined_pca=combined_pca_plot_params,
                    separated_pca= separated_pca_plot_params)
        """
        default_combined_pca = {
            'n_components': 2,
            'figsize': (10, 7),
            'alpha': 0.5,
            'cmap': 'viridis',
            'marker_size': 50,
            'title': f'Combined PCA {args}'
        }
        default_separated_pca = {
            'n_components': 2,
            'figsize': (10, 7),
            'alpha': 0.5,
            'colors': ['tab:orange', 'tab:green', 'tab:red'],
            'marker_size': 50,
            'title': f'Separated PCA {args}'
        }

        if combined_pca is not None:
            default_combined_pca.update(combined_pca)
        if separated_pca is not None:
            default_separated_pca.update(separated_pca)

        num_clusters = len(args)
        flattened_distance_array, linkage_matrix = self.calc_XZ(*args, methodd=method, metricc=metric)

        threshold = linkage_matrix[-num_clusters, 2]
        print(f"Threshold for {num_clusters} clusters: {threshold}")
        fclust = fcluster(linkage_matrix, t=threshold, criterion='distance')
        np.savetxt(os.path.join(self.outPath, f"fclust_{outputFileName}.txt"), fclust)

        pca = PCA(n_components=default_combined_pca['n_components'])
        principalComponents = pca.fit_transform(flattened_distance_array)
        principalDF = pd.DataFrame(data=principalComponents, columns=[f'PC{i+1}' for i in range(default_combined_pca['n_components'])])
        principalDF.to_csv(os.path.join(self.outPath, f'balance_{outputFileName}.csv'))
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

        # Plot the combined PCA results
        if default_combined_pca['n_components'] == 2:
            fig, ax = plt.subplots(1, 1, figsize=default_combined_pca['figsize'])
            scatter = ax.scatter(principalDF["PC1"], principalDF["PC2"], c=fclust, alpha=default_combined_pca['alpha'], cmap=default_combined_pca['cmap'], s=default_combined_pca['marker_size'])
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})', fontsize=8)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})', fontsize=8)
            ax.set_title(default_combined_pca['title'])
            cbar = plt.colorbar(scatter)
            ticks = np.arange(1, num_clusters + 1)
            cbar.set_ticks(ticks)
            plt.savefig(os.path.join(self.outPath, f'PCAs_{outputPlotName}'))
            if show:
                plt.show()
            plt.close()
        
        if default_combined_pca['n_components'] == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=default_combined_pca['figsize'])
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(principalDF["PC1"], principalDF["PC2"], principalDF["PC3"], c=fclust, alpha=default_combined_pca['alpha'], cmap=default_combined_pca['cmap'], s=default_combined_pca['marker_size'])
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})', fontsize=8)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})', fontsize=8)
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2f})', fontsize=8)
            ax.set_title(default_combined_pca['title'])

            cbar = plt.colorbar(scatter)
            ticks = np.arange(1, num_clusters + 1)
            cbar.set_ticks(ticks)
            plt.savefig(os.path.join(self.outPath, f'PCAs_{outputPlotName}'))
            if show:
                plt.show()
            plt.close()

        # Plot the separated PCA results
        if default_separated_pca['n_components'] == 2:
            sizes = {label: self.datasets[label]['distance_array'].shape[0] for label in args}
            
            fig, ax = plt.subplots(1, 1, figsize=default_separated_pca['figsize'])
            start = 0
            for idx, label in enumerate(args):
                end = start + sizes[label]
                scatter = ax.scatter(principalDF["PC1"][start:end], principalDF["PC2"][start:end], alpha=default_separated_pca['alpha'], c=default_separated_pca['colors'][idx], label=label, s=default_separated_pca['marker_size'])
                start = end
            ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.2f})", fontsize=8)
            ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.2f})", fontsize=8)
            ax.set_title(default_separated_pca['title'])
            ax.legend(loc='lower left', fontsize=8, ncol=2)
            cbar = plt.colorbar(scatter)
            plt.savefig(os.path.join(self.outPath, f'PCAs_per_ensemble_{outputPlotName}'))
            if show:
                plt.show()
            plt.close()

        if default_separated_pca['n_components'] == 3:
            sizes = {label: self.datasets[label]['distance_array'].shape[0] for label in args}
            
            fig = plt.figure(figsize=default_separated_pca['figsize'])
            ax = fig.add_subplot(111, projection='3d')
            start = 0
            for idx, label in enumerate(args):
                end = start + sizes[label]
                scatter = ax.scatter(principalDF["PC1"][start:end], principalDF["PC2"][start:end], principalDF["PC3"][start:end], alpha=default_separated_pca['alpha'], c=default_separated_pca['colors'][idx], label=label, s=default_separated_pca['marker_size'])
                start = end
            ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.2f})", fontsize=8)
            ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.2f})", fontsize=8)
            ax.set_zlabel(f"PC 3 ({pca.explained_variance_ratio_[2]:.2f})", fontsize=8)
            cbar = plt.colorbar(scatter)

            ax.set_title(default_separated_pca['title'])
            ax.legend(loc='upper left', fontsize=8, ncol=2)
            plt.savefig(os.path.join(self.outPath, f'PCAs_per_ensemble_{outputPlotName}'))
            if show:
                plt.show()
            plt.close()
        
        

    def tsne_plot(self, *args, show=True, outplotName='TSNE.png', tsneParams=None, plotParams=None, sample_size=5000, num_clusters=None, method='weighted'):
        if num_clusters is None:
            num_clusters = len(args)
            
        default_tsne_params = {
            'n_components': 2,
            'verbose': 1,
            'max_iter': 800,
            'metric': 'euclidean',
        }
        
        default_plot_params = {
            'figsize': (6, 4),
            'cmap': 'viridis',
            'x_label': 't-SNE1',
            'y_label': 't-SNE2',
            'z_label': 't-SNE3',
            'size':50,
            'title':f'TSNE {args}'
        }
        
        if tsneParams is not None:
            default_tsne_params.update(tsneParams)
        if plotParams is not None:
            default_plot_params.update(plotParams)
        
        # Obtain the flattened distance array and linkage matrix from the cache if available
        X, Z = self.calc_XZ(*args, methodd=method, metricc=default_tsne_params['metric'])
        
        # Downsample the data to make it more manageable
        if X.shape[0] > sample_size:
            X = resample(X, n_samples=sample_size, random_state=42)
            
        # Dynamically set perplexity based on the number of samples
        n_samples = X.shape[0]
        perplexity = min(30, max(5, n_samples // 10))
        
        # add perplexity to default_tsne_params
        default_tsne_params['perplexity'] = perplexity  
             
        threshold = Z[-num_clusters, 2]
        print(f"Threshold for {num_clusters} clusters: {threshold}")
        fclust = fcluster(Z, t=threshold, criterion='distance')
        del Z  # Free up memory
        

        # Apply t-SNE 
        tsne = TSNE(**default_tsne_params)
        tsne_res = tsne.fit_transform(X)

        # Save results to a DataFrame and CSV
        tsneDf = pd.DataFrame(data=tsne_res, columns=[f't-SNE{i+1}' for i in range(default_tsne_params['n_components'])])
        tsneDf.to_csv(f'{self.outPath}/tnse_results.csv', index=False)

        # Plot the results
        if default_tsne_params['n_components'] == 2:
            fig, ax = plt.subplots(1, 1, figsize=default_plot_params['figsize'])
            scatter = ax.scatter(tsne_res[:, 0], tsne_res[:, 1], c=fclust, alpha=0.5, s=default_plot_params['size'], cmap=default_plot_params['cmap'])
            ax.set_xlabel(default_plot_params['x_label'])
            ax.set_ylabel(default_plot_params['y_label'])
            ticks = np.arange(1, num_clusters + 1)
            cbar = plt.colorbar(scatter)
            plt.title(default_plot_params['title'])
            cbar.set_ticks(ticks)
            fig.savefig(f'{self.outPath}/{outplotName}')
            if show:
                plt.show()
            plt.close()
        

        if default_tsne_params['n_components'] == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(tsne_res[:, 0], tsne_res[:, 1], tsne_res[:, 2], c=fclust, alpha=0.5, cmap=default_plot_params['cmap'], s=default_plot_params['size'])
            ax.set_xlabel(default_plot_params['x_label'])
            ax.set_ylabel(default_plot_params['y_label'])
            ax.set_zlabel(default_plot_params['z_label'])
            ticks = np.arange(1, num_clusters + 1)
            cbar = plt.colorbar(scatter)
            cbar.set_ticks(ticks)
            fig.savefig(f'{self.outPath}/{outplotName}')
            plt.title(default_plot_params['title'])
            if show:
                plt.show()
            plt.close()



            
    """====================================================================== UTILITIES ===================================================================================="""

    
    def getHiCData_simulation(self,filepath):
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
        
    def getHiCData_experiment(self, filepath, cutoff=0.0, norm="max"):
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
    
    def calc_XZ(self, *args, methodd='weighted', metricc='euclidean'):
        key = tuple(sorted(args))
        key = key + (methodd, metricc)
        if key in self.cache:
            return self.cache[key]['X'], self.cache[key]['Z']
        
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
        flat_euclid_dist_map = {label: [flat_euclid_dist_map[label][val][np.triu_indices_from(flat_euclid_dist_map[label][val], k=1)].flatten()
                                        for val in range(len(flat_euclid_dist_map[label]))]
                                for label in args}
        
        # Make it into a 1D vertical array
        X = np.vstack([item for sublist in flat_euclid_dist_map.values() for item in sublist])
        print(f"Flattened distance array has shape: {X.shape}")
        
        def calcQ(r1, r2):
            shape = X.shape
            sigma = 2 * np.ones(shape=shape[1])
            return np.exp(-(r1 - r2) ** 2 / (sigma ** 2)).mean()
        
        functionPTR={'calcQ':calcQ, 'euclidean':'euclidean'}        
        
        # Create the linkage matrix
        Z = linkage(X, method=methodd, metric=functionPTR[metricc])
        
        self.cache[key] = {'X':X, 'Z': Z}
        
        
        return X, Z 


                
        
        





    
    
