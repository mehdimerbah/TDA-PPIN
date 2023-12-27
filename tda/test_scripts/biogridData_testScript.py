#!/usr/bin/env python3 




# %% 
# # Import packages and required functions
print("Importing packages and required libraries...")
import os
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
degree_dist_dir = os.path.join(os.path.dirname(__file__), 'biogrid_data/plots/degree_dist/')
if not os.path.isdir(degree_dist_dir):
    os.makedirs(degree_dist_dir)

barcode_diags_dir = os.path.join(os.path.dirname(__file__), 'biogrid_data/plots/barcode_diags/')
if not os.path.isdir(barcode_diags_dir):
    os.makedirs(barcode_diags_dir)

## Let's define the persistence diagrams directory to keep this neat
pers_diags_dir = os.path.join(os.path.dirname(__file__), 'biogrid_data/plots/pers_diags/')
if not os.path.isdir(pers_diags_dir):
    os.makedirs(pers_diags_dir)



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
dist_mat_diags_ripser = ripser.ripser(pn_dist_mat, distance_matrix=True, maxdim=3)['dgms']
plot_diagrams(dist_mat_diags_ripser)
plt.savefig(pers_diags_dir+'dist_mat_diags_ripser.png')
plt.close()

# %%
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
