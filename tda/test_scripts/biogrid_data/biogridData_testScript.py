#!/usr/bin/env python3 




# %% 
# # Import packages and required functions
print("Importing packages and required libraries...")
import os
import sys
import warnings
import pandas as pd
import numpy as np
import ripser
from persim import plot_diagrams
import networkx as nwx
import matplotlib.pyplot as plt
import matplotlib
import random
from sklearn.cluster import DBSCAN
import gudhi as gd
import gudhi.representations
import math
import itertools

# %% 
# # Setting directories for output files
print("Creating the appropriate directories for script output...")
degree_dist_dir = os.path.join(os.path.dirname(__file__), 'plots/degree_dist/')
if not os.path.isdir(degree_dist_dir):
    os.makedirs(degree_dist_dir)

barcode_diags_dir = os.path.join(os.path.dirname(__file__), 'plots/barcode_diags/')
if not os.path.isdir(barcode_diags_dir):
    os.makedirs(barcode_diags_dir)

## Let's define the persistence diagrams directory to keep this neat
pers_diags_dir = os.path.join(os.path.dirname(__file__), 'plots/pers_diags/')
if not os.path.isdir(pers_diags_dir):
    os.makedirs(pers_diags_dir)

## Creating a directory for the PH reuslts
ph_results_dir = os.path.join(os.path.dirname(__file__), 'results/')
if not os.path.isdir(ph_results_dir):
   os.makedirs(ph_results_dir)


# %%
# # Import data
print("Importing the data for Biogird Network and CORUM Complexes...")
ppi_df = pd.read_table("../data/Human_PPI_Network.txt", header=None)
ppi_df.columns = ["ProteinA", "ProteinB", "SemSim"]
complexes_list = []
with open("../data/CORUM_Human_Complexes.txt") as complexes:
  for line in complexes:
    line = line.strip()
    temp_list = list(line.split("\t"))
    complexes_list.append(temp_list)

# %%
# # Data Exploration
# ## CORUM Data
print("Checking the CORUM data and enumerating complexes...")
complexes_dict = {}
complexes_single_proteins = []
for idx, cmplx in enumerate(complexes_list):
  for protein in cmplx:
    if protein not in complexes_single_proteins:
      complexes_single_proteins.append(protein)

  complexes_dict[idx] = cmplx

print('There are %d complexes in the CORUM Data set.' % len(complexes_dict))
print('There are %d individual proteins in the complexes' % len(complexes_single_proteins))


# ## Biogrid Data
"""Extracting unique proteins"""

ppi_single_proteins_m = ppi_df['ProteinA'].unique()
ppi_single_proteins_n = ppi_df['ProteinB'].unique()
ppi_single_proteins = set(ppi_single_proteins_m).union(set(ppi_single_proteins_n))
n_ppi_proteins =  len(ppi_single_proteins)

"""Create the network with networkX package"""

print("Creating NetworkX PPI network for the entire dataset.")
biogrid_protein_net = nwx.from_pandas_edgelist(
        ppi_df,
        source='ProteinA',
        target='ProteinB',
        edge_attr='SemSim'
    )


# ## Node Degree ditribution across the network.

def viz_degree(network_graph):
  degree_nodes = {}
  for p, d in network_graph.degree():
    degree_nodes[p] = d

  ## This gives that there are nodes with degree > 900
  sorted_node_degrees = dict(sorted(degree_nodes.items(), key=lambda item: item[1],  reverse=True))

  ## Let's visualize the distribution
  viz_degree = {degree: 0 for degree in degree_nodes.values()}
  for degree in degree_nodes.values():
    viz_degree[degree]+=1
  degree_count_pairs = sorted(viz_degree.items())
  x, y = zip(*degree_count_pairs) # unpack a list of pairs into two tuples
  plt.plot(x, y)
  plt.xlabel('Node Degree')
  plt.ylabel('Protein Count')
  plt.title('Node Degree Distribution Biogrid PPI Network')
  plt.savefig(degree_dist_dir+'PPI_Net_node_degree_distribution.png')

viz_degree(biogrid_protein_net)

# %% 
# # Persistent Homology

"""This is a customized function I wrote to plot the barcode specifically for Ripser package Persistence Diagrams"""
def plot_barcode(diag, dim, plot_title, **kwargs):
    diag_dim = diag[dim]
    birth = diag_dim[:, 0]; death = diag_dim[:, 1]
    finite_bars = death[death != np.inf]
    if len(finite_bars) > 0:
        inf_end = 2 * max(finite_bars)
    else:
        inf_end = 2
    death[death == np.inf] = inf_end
    plt.figure(figsize=kwargs.get('figsize', (10, 5)))
    for i, (b, d) in enumerate(zip(birth, death)):
        if d == inf_end:
            plt.plot([b, d], [i, i], color='k', lw=kwargs.get('linewidth', 2))
        else:
            plt.plot([b, d], [i, i], color=kwargs.get('color', 'b'), lw=kwargs.get('linewidth', 2))
    plt.title(kwargs.get('title', f'Pers Barcode Dim: {dim}'))
    plt.xlabel(kwargs.get('xlabel', 'Filtration Value'))
    plt.yticks([])
    plt.tight_layout()

    plt_name_format = f'{plot_title}.png'
    plt_path = barcode_diags_dir + plt_name_format
    plt.savefig(plt_path)
    plt.close()

print("Creating Adjacency Matrix fro the protein network...")
pn_adjacency = nwx.adjacency_matrix(biogrid_protein_net).toarray()
np.fill_diagonal(pn_adjacency, 1)
print("Creating Corr-Distance Matrix from Adjacency Matrix...")
pn_dist_mat = 1 - pn_adjacency

"""Applying Vietoris Rips Filtration on the network"""

# %%
## PH with correlation distance matrix
print("Computing Ripser Persistent Homology on Corr-Dist matrix...")
dist_mat_diags_ripser = ripser.ripser(pn_dist_mat, distance_matrix=True, maxdim=3)['dgms']
plot_diagrams(dist_mat_diags_ripser)
plt.savefig(pers_diags_dir+'dist_mat_diags_ripser.png')
plt.close()

# %%
print("Saving Ripser persistence diagrams...")
plot_diagrams(dist_mat_diags_ripser, plot_only=[0], title='Corr-Dist Pers Diagram Dim 0')
plt.savefig(pers_diags_dir+'dist_mat_diags_ripser_Dim0.png')
plt.close()
plot_barcode(dist_mat_diags_ripser, 0, 'Corr-Dist Barcode Diag Dim 0')

# %%
plot_diagrams(dist_mat_diags_ripser, plot_only=[1], title='Corr-Dist Pers Diagram Dim 1')
plt.savefig(pers_diags_dir+'dist_mat_diags_ripser_Dim1.png')
plt.close()
plot_barcode(dist_mat_diags_ripser, 1, 'Corr-Dist Barcode Diag Dim 1')

# %%
plot_diagrams(dist_mat_diags_ripser, plot_only=[2], title='Corr-Dist Pers Diagram Dim 2')
plt.savefig(pers_diags_dir+'dist_mat_diags_ripser_Dim2.png')
plt.close()
plot_barcode(dist_mat_diags_ripser, 2, 'Corr-Dist Barcode Diag Dim 2')

# ## Gudhi Package Rips Complex Construction

print("Constructing Rips Complex from Corr-Dist matrix...")
rips_complex = gd.RipsComplex(distance_matrix=pn_dist_mat, max_edge_length=1.0)


## We now build a simplex tree to store the simplices
print("Building Rips simplex tree...")
rips_simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)


## Check how the complex looks like
result_str = 'Rips complex is of dimension ' + repr(rips_simplex_tree.dimension()) + ' - ' + \
    repr(rips_simplex_tree.num_simplices()) + ' simplices - ' + \
    repr(rips_simplex_tree.num_vertices()) + ' vertices.\n'


with open('gudhi_VR_PH_results.txt', 'at') as results_file:
   results_file.write(result_str)
   results_file.close()

## Apply Persistence
print("Computing persistence on Rips Simplex Tree...")
persistence = rips_simplex_tree.persistence(min_persistence=-1, persistence_dim_max=True)

# Generate persistence diagrams
print("Saving Rips Simplex Tree persistence diagrams...")
diagrams = gd.plot_persistence_diagram(persistence, max_intervals=4000000, title='Corr-Dist Mat Pers Diag')
plt.savefig(pers_diags_dir+'gudhi_VR_pers_diagrams.png')
plt.close()


# %% 
# Extracting the relavant persistent features 
def extract_pers_feat_in_dim(persistence_list, dim, birth_max, death_min):
    pers_feat = []
    for pt in persistence_list:
        pt_dim = pt[0]
        birth = pt[1][0]
        death = pt[1][1]
        if pt_dim == dim and birth < birth_max and death > death_min:
            pers_feat.append(list(pt[1]))
    return pers_feat

# %%
# Creating embeddings into persistence landscapes for persistent homology features

pers_LS = gd.representations.Landscape(num_landscapes = 5).fit_transform(dist_mat_diags_ripser[1])




print("Script finished executing!")
sys.exit()

