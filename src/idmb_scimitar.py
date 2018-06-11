"""
Imputated Diffusion Map and Branching SCIMITAR for Trajectory Inference in scRNA-seq data (IDMB-SCIMITAR)
Contact info: lseninge@ucsc.edu 

This script aims at imputing missing data from preprocessed scRNA-seq dataframe using MAGIC(KrishnaswamyLab)
and uses this imputed data to compute a Diffusion Map, used by branching SCIMITAR (Pablo Cordero)
to infer a branching trajectory in the data.

Recommended usage: Run on subset of your data or individual clusters; SCIMITAR may fails to infer trajectory
if imputed diffusion map has no 'trajectory-like' structure in it.

Example usage:
import idmb_scimitar

#read data with cells as rows, genes as columns
my_data_clust1=pd.pd.read_csv('/path/to/my/data_clust1.csv', sep=',', index_col=0)

#To get imputed diffusion map 
imp_df=imputation(my_data_clust1)

#(Recommended before running Trajectory Inference to identify clusters with trajectory-like structure)
map_df=get_imputed_diffmap(imp_df)
plot_imputed_diffmap(map_df, data=imp_df )

#To (directly) get scimitar branching trajectory
model=scimitar_model(imp_df, n_nodes=30)



See help(idmb_scimitar)
"""



import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.manifold import SpectralEmbedding
from magic.MAGIC_core import magic
from scipy.stats import entropy
import matplotlib.pyplot as plt
from scimitar.branching import BranchedEmbeddedGaussians as BEG
from scipy.spatial.distance import cdist
import seaborn as sns


def imputation( dataframe , frac=0.1):    
    """
    This function impute values using MAGIC package.
    Also filter out genes with less than X% non-zero expression. Default 10%.
    :param dataframe: dataframe dataframe with cells as rows and genes as columns
    :return magic_data: filtered, imputed dataframe
    """
    magic_data = pd.DataFrame(magic(dataframe.values),
                                         index=dataframe.index,
                                         columns=dataframe.columns)
    magic_data.fillna(0, inplace=True)
    magic_data = filter_percent_expressed(magic_data, frac)
    plt.close()
    return magic_data


def get_imputed_diffmap(imputed_df):
    """
    This function compute 2D Diffusion Map coordinates based on imputed data matrix.
    :param dataframe: dataframe returned by imputation(); with cells as rows and genes as columns
    
    :return diff_map: imputed, 2D diffusion map coordinates for each cell
    """
    a_rbf_mtx = _adaptive_rbf_matrix(imputed_df)
    embedding = SpectralEmbedding(n_components=2, affinity='precomputed')
    diff_map = embedding.fit_transform(a_rbf_mtx)
    diff_map = pd.DataFrame(diff_map , index=imputed_df.index , columns=['diff_map_1', 'diff_map_2'])
    return diff_map

#Maybe add possibility to color by categorical in plot_imputed_diffmap()
def plot_imputed_diffmap(coord, data=None , colors=None):
    """
    Scatter plot representation of the imputed diffusion map.
    Possibility to color by: Gene expression in original data, or entropy. Default = blue
    :param coord: output from get_imputed_diffmap()
    :param data: imputed dataframe from imputation(); Default=None
    :param color: gene name as a string to use to color cells in 2D diffusion map;
                  if 'entropy' is passed, calls _entropy_data function to color
                  by entropy (seems correlated with differentiation process )
                  
    :return : A 2D plot of the data
    """
    title='Imputed Diffusion Map'
    use_color=colors
    if colors is None:
        colors='blue'
        cmap=None
        plt.scatter(coord.values[:, 0], coord.values[:, 1],
                        s=100, linewidth=0., alpha=0.5,
                        c=colors, cmap=cmap)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
    elif colors != None:
        
        if data is None:
            return ('To color by gene or entropy, you need to provide the original data!')
        
        elif colors == 'entropy':
            data_entropies=data.apply(_entropy_data, axis=1).values
            cmap = plt.get_cmap('plasma')
            colors=data_entropies
            plt.scatter(coord.values[:, 0], coord.values[:, 1],
                        s=100, linewidth=0., alpha=0.5,
                        c=colors, cmap=cmap)
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            
        elif colors not in list(data):
            return ('Your input gene is not in the original data, or was filtered out during imputation!')
        
        else:
            cmap = plt.get_cmap('plasma')
            colors=data[colors]
            plt.scatter(coord.values[:, 0], coord.values[:, 1],
                        s=100, linewidth=0., alpha=0.5,
                        c=colors, cmap=cmap)
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            
    if use_color == 'entropy' or use_color in list(data):
        plt.colorbar().set_label(use_color, labelpad=25, rotation=270, weight='bold')
    plt.title(title)
    plt.show()


def scimitar_model(data, 
                 n_nodes=100, 
                 n_components=2, 
                 npcs=10, 
                 sigma=0.1, 
                 gamma=0.5, 
                 n_neighbors=30, 
                 just_visualize=False,
                  colors=None):
    """
    This function calls BranchedEmbeddedGaussians from scimitar.branching to find trajectory in the diffusion map
    :param data: imputed dataframe from imputation()
    :param n_nodes: number of nodes in the trajectory
    :param n_components: dimensions to use
    :param npcs: number of PCs to use in the analysis
    :param colors: gene name as a string to use to color cells in 2D diffusion map;
                  if 'entropy' is passed, calls _entropy_data function to color
                  by entropy (seems correlated with differentiation process )
    
    :return result: tuple containing the scimitar model, probabilities of assignment to each cell to nodes
                    and node assignment
    """
    
    model = BEG(n_nodes=n_nodes, 
                npcs=npcs, 
                max_iter=10, 
                sigma=sigma,
                gamma=gamma,
                n_neighbors=n_neighbors,
                embedding_dims=n_components,
                just_tree=just_visualize)
    model.fit(data.values)
    
    assignments = np.zeros([data.shape[0]]) - 1
    neg_log_probs = cdist(model._embedding_tx, model.node_positions) 
    for i in range(data.shape[0]):    
        assignments[i] = np.argmin(neg_log_probs[i, :])
    
    num_leafs = len([n for n in model.graph.nodes() if model.graph.degree(n) == 1])
    print('Number of branches: %s' % (num_leafs - 1))
        
    _plot_scimitar_model(model, model._embedding_tx, colors, data)
    title=colors
    plt.title('Scimitar Branching Tree with '+str(n_nodes)+' nodes')
    plt.figure()
    plt.show()
    
    result = model, neg_log_probs, assignments
    return result



def _filter_percent_expressed(df, frac=0.1):
    """
    Filter out genes with less than X% non-zero expression. Default 10%.
    :param df: dataframe containing data (preprocessed matrix)
    
    :return df[selected_genes]: filtered dataframe
    """
    selected_genes = df.columns[((df != 0).mean(axis=0) > frac).values]
    return df[selected_genes]
    
    
def _entropy_data(data):
    """
    This function computes entropy of information of input data
    :param data: dataframe (with cells as rows and genes as columns)
    
    :return entropy values of input
    """
    bin_fracs = np.histogram(data.values, bins=50, density=True)[0]
    entropies=entropy(bin_fracs)
    return entropies


def _adaptive_rbf_matrix(data_array, n_neighbors=30, scale=1.0):
    n_samples = data_array.shape[0]
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(data_array)
    A = pairwise_distances(data_array, metric='l2')
    
    n_distances = np.reshape(nn.kneighbors(data_array)[1][:, -1], (n_samples, 1))
    S = np.dot(n_distances, n_distances.T) / A.mean()
    A = np.exp(-scale * (A + 1)/(S + 1))
    return A

#Add colors to the plot if users wants to plot with gene expression, or entropy (cf plot_diff_map())
def _plot_scimitar_model(pt, tx, colors , data):
    """
    This function plots the output of scimitar_model() to visualize trajectory
    :param pt: the scimitar object from BranchedEmbeddedGaussians analysis
    :param tx: the diffusion map coordinates
    :param :color used to colors cells in the plot (passed in scimitar_model function)
    :param data: imputed, filtered dataframe used in scimitar_model 
    
    :return A plot with cells in the diffusion map and infered branching trajectory
    """
    use_color=colors
    
    #Plot
    if colors is None:
        colors='b'
        cmap=None
    
    elif colors != None:
        if colors == 'entropy':
            data_entropies=data.apply(_entropy_data, axis=1).values
            cmap = plt.get_cmap('plasma')
            colors=data_entropies
        elif colors not in list(data):
            print ('\nYour input gene is not in the original data, or was filtered out during imputation!\nUsing Colors=None')
            colors='r'
            cmap=None
        else:
            cmap = plt.get_cmap('plasma')
            colors=data[colors]
    

    plt.scatter(tx[:, 0], tx[:, 1], c=colors, cmap=cmap, alpha=0.1, s=200)
    
    if use_color =='entropy' or use_color in list(data):
        plt.colorbar().set_label(use_color, labelpad=25, rotation=270, weight='bold')

    plt.scatter(pt.node_positions[:, 0], pt.node_positions[:, 1], alpha=0.5, c='k', s=100)
    X, Y = [pt.node_positions[i, 0] for i in range(pt.node_positions.shape[0])], [pt.node_positions[i, 1] 
                                                   for i in range(pt.node_positions.shape[0])]
    for i, j in pt.graph.edges():
        plt.plot([X[i], X[j]], [Y[i], Y[j]],'k-', zorder=1, linewidth=1)
    
