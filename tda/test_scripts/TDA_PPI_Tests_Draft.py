#!/usr/bin/env python3 

"""
This is a draft document to edit sub-scripts to run later, Do not run this independently.

"""

## ------------------------------------------------------------------------------ ##


# %% [markdown]
# ### Exploring Simplices in Simplex Tree

# %%
smplx_tree_gen = simplex_tree.get_filtration()

# %%
for smplx in smplx_tree_gen:
    if(len(smplx[0]) == 1):
        print(smplx)

#if len(smplx[0] == 1):

# %%
simplex_tree.dimension()

# %%
simplex_tree.num_simplices()

# %%
simplex_tree.num_vertices()

# %% [markdown]
# ### Feature Extraction and Clustering

# %%
## death values of np.inf would not give us a measure to cluster them with DBSCAN, so let's set a cap death time for those to a max value of 1
for i, pt in enumerate(persistence):
    if pt[1][1] == np.inf:
        pt = list(pt)
        pt[1] = list(pt[1])
        pt[1][1] = 1
        pt[1] = tuple(pt[1])
        persistence[i] = tuple(pt)
        #print(pt)

# %%

# Extract features from the persistence diagrams
# I will use just the birth and death times of the topological features for now and see how DBSCAN would perfrom in the clustering
features = np.array([[pt[1][0], pt[1][1]] for pt in persistence])


# Cluster the proteins into complexes using DBSCAN
dbscan = DBSCAN(eps=0.1, min_samples=5)
dbscan.fit(features)

# Get the predicted labels (complexes)
predicted_labels = dbscan.labels_


## ------------------------------------------------------------------------------- ##


# %% [markdown]
# ## Exploring Persistent Features

## %%
def extract_pers_feat_in_dim(persistence_list, dim, birth_max, death_min):
    pers_feat = []
    for pt in persistence_list:
        pt_dim = pt[0]
        birth = pt[1][0]
        death = pt[1][1]
        if pt_dim == dim and birth < birth_max and death > death_min:
            pers_feat.append(list(pt[1]))
    return pers_feat

## %%
dim_1_pers_features = extract_pers_feat_in_dim(persistence, 1, 0.62, 0.55)

## %%
for filtered_value in rips_simplex_tree.get_filtration():
    dim = len(filtered_value[0])
    if dim == 2 and filtered_value[1] in dim_1_pers_features:
        print(filtered_value)

## %%
rips_smplx_combs = list(itertools.combinations(complex_3, 2))

# %%
idx_rips_smplx_combs = []
for comb in rips_smplx_combs:
    tmp = []
    for protein in comb:
        tmp.append(int(protein.strip("Protein_"))-1)
    idx_rips_smplx_combs.append(tmp)

# %%
rips_smplx_perms = list(itertools.permutations(complex_3, 2))

# %%
idx_rips_smplx_perms = []
for comb in rips_smplx_perms:
    tmp = []
    for protein in comb:
        tmp.append(int(protein.strip("Protein_"))-1)
    idx_rips_smplx_perms.append(tmp)

# %%
len(idx_rips_smplx_combs)

# %%
len(idx_rips_smplx_perms)

# %%
for filtered_value in rips_simplex_tree.get_filtration():
    if len(filtered_value[0]) == 2:
        if filtered_value[0] in idx_rips_smplx_perms:
            print(filtered_value)

# %%
rips_smplx_pers_intervals_dim2 = rips_simplex_tree.persistence_intervals_in_dimension(1)

# %%
relevant_filtrations = []
for interval in rips_smplx_pers_intervals_dim2:
    if interval[1] - interval[0] > 0.5:
        print(interval)
        relevant_filtrations.append(interval[0])

# %%
persistent_simplicial_cmplxs = []
for filtered_value in rips_simplex_tree.get_filtration():
    if len(filtered_value[0]) == 2 and filtered_value[1] in relevant_filtrations:
        print(filtered_value)
        persistent_simplicial_cmplxs.append(filtered_value)

