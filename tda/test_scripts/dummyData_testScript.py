#!/usr/bin/env python3 




# %% 
# # Import packages and required functions
import os
import pandas as pd
import numpy as np
import ripser
from persim import plot_diagrams
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import random
from sklearn.cluster import DBSCAN
import gudhi as gd
import gudhi.representations
import math
import itertools

# %% [markdown]
# # Experimenting with Randomized Synthetic Network


## Set barcode directories to save plots

degree_dist_dir = os.path.join(os.path.dirname(__file__), 'plots/dummy_data/degree_dist/')
if not os.path.isdir(degree_dist_dir):
    os.makedirs(degree_dist_dir)

barcode_diags_dir = os.path.join(os.path.dirname(__file__), 'plots/dummy_data/barcode_diags/')
if not os.path.isdir(barcode_diags_dir):
    os.makedirs(barcode_diags_dir)

## Let's define the persistence diagrams directory to keep this neat
pers_diags_dir = os.path.join(os.path.dirname(__file__), 'plots/dummy_data/pers_diags/')
if not os.path.isdir(pers_diags_dir):
    os.makedirs(pers_diags_dir)

# %% 
## Define Barcode Function for later use
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

# %% [markdown]
# ## Creating a Randomized PPI Network 

# %% 
## Create Network
"""RANDOMIZED Network construction based on a given edge presence probability"""

# Initializing empty graph with the Networkx Package
protein_network = nx.Graph()

# Adding 100 proteins as nodes
proteins = [f"Protein_{i}" for i in range(1, 101)]
protein_network.add_nodes_from(proteins)

# Adding some randomization by assigning a probability parameter for edge creation
edge_probability = 0.20

# Assigning random weights to edges with given probability
for i, protein1 in enumerate(proteins):
    for j, protein2 in enumerate(proteins):
        if i < j and random.random() < edge_probability:  # Ensuring to add only one side of the edge to avoid duplicates and self-loops
            # Assigning random weights for edges
            protein_network.add_edge(protein1, protein2, weight=random.uniform(0, 0.7))


## %% Adding Complexes
# Creating protein complexes (connecting nodes within complexes with higher weights)
complex_1 = random.sample(proteins, 10)
complex_2 = random.sample(proteins, 7)
complex_3 = random.sample(proteins, 5)

# Assigning higher weights to edges between proteins that belong in the complexes
for complex_proteins in [complex_1, complex_2, complex_3]:
    for i in range(len(complex_proteins)):
        for j in range(i+1, len(complex_proteins)):
            if protein_network.has_edge(complex_proteins[i], complex_proteins[j]):
                ## For each protein belonging to a complex double the edge weight to indicate stronger interaction
                if protein_network[complex_proteins[i]][complex_proteins[j]]['weight'] >= 0.5:
                    protein_network[complex_proteins[i]][complex_proteins[j]]['weight'] = 0.95
                else:
                    protein_network[complex_proteins[i]][complex_proteins[j]]['weight'] *= 2  




# %% 
## Node Degree Viualization
""" The following is just to check the distribution of the node degrees. As it seems like there are highly central nodes """
degree_nodes = {}
for p, d in protein_network.degree():
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

plt.savefig(str(degree_dist_dir)+'degree_dist.png')
plt.close()





# %% [markdown]
# # Applying Persistent Homology
""" This is where we apply persistent homology on the dataset, we use both ripser and gudhi to extract the topological features as both libraries are useful in their own regard"""

# %% [markdown]
# ## Using the Adjacency Matrix
"""Using the adjacency matrix from the networkX package with the ripser package defining the distance_matrix pararm as 'False'"""

# %%
## Extract the adjacency matrix from the NetworkX package and fill the diagonal to indicate node self-connection
pn_adjacency = nx.adjacency_matrix(protein_network).toarray()
np.fill_diagonal(pn_adjacency, 1)


# %%
## Run the Vietoris-Rips filtrations on the adjacency matrix (maximum homology dimension is set to 3)
## This would not make too much sense as we should be feeding a distance measure between the nodes. Although we have a square matrix we are not on a metric space.
pn_diagrams = ripser.ripser(pn_adjacency, distance_matrix=False, maxdim=3)['dgms']

# %% [markdown]
# ## Using Correlation Distance Matrix
"""The reasoning here is that if the score value for the proteins measures semantic similarity, converting it 
to a dissimilarity measure would be a distance measure. as such protein-pairs with low scores would be "closer" to eachother."""
np.fill_diagonal(pn_adjacency, 1)
pn_distance_mat = 1 - pn_adjacency

# %%
## PH with correlation distance matrix
dist_mat_diags_ripser = ripser.ripser(pn_distance_mat, distance_matrix=True, maxdim=3)['dgms']
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

# %% [markdown]
# ## GUDHI Persistent Homology Analysis and Clustering

# %% [markdown]
# ### Simplex Tree Construction

# %%
## Let's store the proteins in an indexed dictionary so we could track them back when building the simplex tree
proteins_dict = {protein: idx for idx, protein in enumerate(proteins)}

# %%
# Given protein-protein interaction network we can start by creating a simplex tree that includes all 0-simplices (nodes)

# Construct a simplex tree from the network
simplex_tree = gd.SimplexTree()

for edge in protein_network.edges(data=True):
    node1, node2, weight = edge
    ## Get protein index from dict to map it back and feed it into the simplex tree
    node1_idx = proteins_dict[node1]
    node2_idx = proteins_dict[node2]
    simplex_tree.insert([node1_idx, node2_idx], filtration=weight['weight'])



# %% [markdown]
# ### Persistent Homology

# %%
# Compute persistence diagrams
## NOTE: min_persistence is set to -1 to view all the simplex values (Include all 0-simplices)

persistence = simplex_tree.persistence(min_persistence=-1, persistence_dim_max=False)

# Generate persistence diagrams
diagrams = gd.plot_persistence_diagram(persistence)
plt.savefig(pers_diags_dir+'gudhi_pers_diag.png')
barcode = gd.plot_persistence_barcode(persistence)
plt.savefig(barcode_diags_dir+'gudhi_barcode_diag.png')
density = gd.plot_persistence_density(persistence)
plt.savefig('plots/dummy_data/guhdi_density_plot.png')

# %% [markdown]
# ## GUDHI RipsComplex Construction

# %%
## Here we use the Gudhi library to build the Rips complex and apply the homology.
## We can build the Rips simplicial complex by using a distance matrix. So I just plugged in the correlation distance matrix.
rips_complex = gd.RipsComplex(distance_matrix=pn_distance_mat, max_edge_length=1.0)


## We now build a simplex tree to store the simplices
rips_simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)


## Check how the complex looks like
result_str = 'Rips complex is of dimension ' + repr(rips_simplex_tree.dimension()) + ' - ' + \
    repr(rips_simplex_tree.num_simplices()) + ' simplices - ' + \
    repr(rips_simplex_tree.num_vertices()) + ' vertices.'

print(result_str)

# %%
persistence = rips_simplex_tree.persistence(min_persistence=-1, persistence_dim_max=True)

# %%
# Generate persistence diagrams
diagrams = gd.plot_persistence_diagram(persistence, max_intervals=4000000)
plt.savefig(pers_diags_dir+'gudhi_VR_pers_diag.png')
plt.close()

