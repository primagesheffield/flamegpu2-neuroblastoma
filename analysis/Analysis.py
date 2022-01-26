#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:45:20 2022

@author: kywertheim
"""

"""
Load the necessary libraries.
"""
import pandas
import numpy
import sklearn.preprocessing as sklprepro
import matplotlib.pyplot as plt
import sklearn.decomposition as skldecomp
import sklearn.cluster as sklcluster
import sklearn.metrics.cluster as sklmetrics
import scipy.cluster.hierarchy as spcluster

"""
Load the dataset.
"""
df_processed = pandas.read_pickle("hetNB_results.pkl")

"""
Exploratory analysis: O2, cellularity, theta_sc, and degdiff.
Conclusion: normalisation is necessary.
"""
df_processed.iloc[:,1:5].describe()
df_processed.iloc[:,1:5].hist()
df_processed.iloc[:,1:5].boxplot()

"""
Exploratory analysis: clones 1 to 6.
Conclusion: log transformation is necessary.
"""
df_processed.iloc[:,5:11].describe()
df_processed.iloc[:,5:11].hist()
df_processed.iloc[:,5:11].boxplot()


"""
Exploratory analysis: clones 7 to 12.
Conclusion: log transformation is necessary.
"""
df_processed.iloc[:,11:17].describe()
df_processed.iloc[:,11:17].hist()
df_processed.iloc[:,11:17].boxplot()

"""
Exploratory analysis: clones 13 to 18.
Conclusion: log transformation is necessary.
"""
df_processed.iloc[:,17:23].describe()
df_processed.iloc[:,17:23].hist()
df_processed.iloc[:,17:23].boxplot()

"""
Exploratory analysis: clones 19 to 24.
Conclusion: log transformation is necessary.
"""
df_processed.iloc[:,23:29].describe()
df_processed.iloc[:,23:29].hist()
df_processed.iloc[:,23:29].boxplot()

"""
Exploratory analysis: initial fraction of cells with MYCN amplification.
Conclusion: normalisation is necessary.
"""
df_processed.iloc[:,29].describe()
df_processed.iloc[:,29].hist()
df_processed.iloc[:,29:30].boxplot()

"""
Exploratory analysis: final fraction of cells with MYCN amplification.
"""
df_processed.iloc[:,30:32].describe()
df_processed.iloc[:,30:32].hist()
df_processed.iloc[:,30:32].boxplot()

"""
Transform the input space.
"""
df_processed.iloc[:,5:29] = df_processed.iloc[:,5:29].apply(lambda xi: numpy.log(xi)) #Clonal fractions require log transformation.
scaler=sklprepro.MinMaxScaler() #After the log transformation, normalise all the features.
df_processed.iloc[:,1:30]=scaler.fit_transform(df_processed.iloc[:,1:30])

"""
Exploratory analysis again.
"""
df_processed.iloc[:,1:5].describe()
df_processed.iloc[:,1:5].hist()
df_processed.iloc[:,1:5].boxplot()

df_processed.iloc[:,5:29].describe()
df_processed.iloc[:,5:29].hist()
df_processed.iloc[:,5:29].boxplot()

df_processed.iloc[:,29].describe()
df_processed.iloc[:,29].hist()
df_processed.iloc[:,29:30].boxplot()

"""
Split the dataset into three categories.
"""
df_processed_R = df_processed[df_processed.R>5]
df_processed_R['Lablel'] = 'R'
df_processed_D = df_processed[df_processed.D>5]
df_processed_D['Lablel'] = 'D'
df_processed_O = df_processed[df_processed.O>5]
df_processed_O['Lablel'] = 'O'

"""
Exploratory analysis.
1. R: 47 cases, D: 1070 cases, and O: 83 cases.
2. Every configuration belongs to one category.
"""
df_processed_R.shape[0]
df_processed_D.shape[0]
df_processed_O.shape[0]
df_processed.shape[0] == df_processed_R.shape[0] + df_processed_D.shape[0] + df_processed_O.shape[0]

"""
Identify MYCN enrichment in the O dataset.
"""
df_processed_O_MYCNenriched = df_processed_O[df_processed_O.MYCN_final_init>1]

"""
Exploratory analysis: MYCN enrichment can be taken for granted in the O dataset.
"""
df_processed_O.shape[0] == df_processed_O_MYCNenriched.shape[0]

"""
Exploratory analysis of the R dataset.
1. Low initial O2 is the key feature in this dataset.
2. Compared to the original dataset, the R dataset contains weak evidence of MYCN
enrichment during the runs. However, a regression incident in a run was automatically
assigned MYCN_final=0 and MYCN_final_init=0.
"""
df_processed_R.iloc[:,1:5].describe()
df_processed_R.iloc[:,1:5].hist()
df_processed_R.iloc[:,1:5].boxplot()

df_processed_R.iloc[:,5:29].describe()
df_processed_R.iloc[:,5:29].hist()
df_processed_R.iloc[:,5:29].boxplot()

df_processed_R.iloc[:,29].describe()
df_processed_R.iloc[:,29].hist()
df_processed_R.iloc[:,29:30].boxplot()

R_inputs = numpy.zeros((2,29)) #Compare the average inputs in the R dataset with those in the original dataset.
R_inputs[0, :] = numpy.array(df_processed.iloc[:,1:30].mean())
R_inputs[1, :] = numpy.array(df_processed_R.iloc[:,1:30].mean())
R_inputs[0]-R_inputs[1]

df_processed_R.iloc[:,30:32].describe()
df_processed_R.iloc[:,30:32].hist()
df_processed_R.iloc[:,30:32].boxplot()

R_outputs = numpy.zeros((2,2))
R_outputs[0, :] = numpy.array(df_processed.iloc[:,30:32].mean())
R_outputs[1, :] = numpy.array(df_processed_R.iloc[:,30:32].mean())
R_outputs[0]-R_outputs[1]

"""
Exploratory analysis of the D dataset.
1. No key features.
"""
df_processed_D.iloc[:,1:5].describe()
df_processed_D.iloc[:,1:5].hist()
df_processed_D.iloc[:,1:5].boxplot()

df_processed_D.iloc[:,5:29].describe()
df_processed_D.iloc[:,5:29].hist()
df_processed_D.iloc[:,5:29].boxplot()

df_processed_D.iloc[:,29].describe()
df_processed_D.iloc[:,29].hist()
df_processed_D.iloc[:,29:30].boxplot()

D_inputs = numpy.zeros((2,29)) #Compare the average inputs in the D dataset with those in the original dataset.
D_inputs[0, :] = numpy.array(df_processed.iloc[:,1:30].mean())
D_inputs[1, :] = numpy.array(df_processed_D.iloc[:,1:30].mean())
D_inputs[0]-D_inputs[1]

df_processed_D.iloc[:,30:32].describe()
df_processed_D.iloc[:,30:32].hist()
df_processed_D.iloc[:,30:32].boxplot()

D_outputs = numpy.zeros((2,2))
D_outputs[0, :] = numpy.array(df_processed.iloc[:,30:32].mean())
D_outputs[1, :] = numpy.array(df_processed_D.iloc[:,30:32].mean())
D_outputs[0]-D_outputs[1]

"""
Exploratory analysis of the O dataset.
1. Low cellularity, theta_sc, and degdiff are the key features in this dataset.
2. In particular, theta_sc is tiny.
3. Compared to the original dataset, the O dataset contains stronger evidence of
MYCN enrichment during the runs. In fact, every MYCN_final_init in the O dataset
is bigger than 1. From this point onwards, df_processed_O_MYCNenriched will be
ignored in this analysis.
"""
df_processed_O.iloc[:,1:5].describe()
df_processed_O.iloc[:,1:5].hist()
df_processed_O.iloc[:,1:5].boxplot()

df_processed_O.iloc[:,5:29].describe()
df_processed_O.iloc[:,5:29].hist()
df_processed_O.iloc[:,5:29].boxplot()

df_processed_O.iloc[:,29].describe()
df_processed_O.iloc[:,29].hist()
df_processed_O.iloc[:,29:30].boxplot()

O_inputs = numpy.zeros((3,29)) #Compare the average inputs in the O dataset, O-MYCNamp dataset, and the original dataset.
O_inputs[0, :] = numpy.array(df_processed.iloc[:,1:30].mean())
O_inputs[1, :] = numpy.array(df_processed_O.iloc[:,1:30].mean())
O_inputs[2, :] = numpy.array(df_processed_O_MYCNenriched.iloc[:,1:30].mean())
O_inputs[0]-O_inputs[1]
O_inputs[0]-O_inputs[2]
O_inputs[1]-O_inputs[2]

df_processed_O.iloc[:,30:32].describe()
df_processed_O.iloc[:,30:32].hist()
df_processed_O.iloc[:,30:32].boxplot()

O_outputs = numpy.zeros((3,2))
O_outputs[0, :] = numpy.array(df_processed.iloc[:,30:32].mean())
O_outputs[1, :] = numpy.array(df_processed_O.iloc[:,30:32].mean())
O_outputs[2, :] = numpy.array(df_processed_O_MYCNenriched.iloc[:,30:32].mean())
O_outputs[0]-O_outputs[1]
O_outputs[0]-O_outputs[2]
O_outputs[1]-O_outputs[2]

"""
Exploratory analysis: the influence of MYCN_init on clonal evolution.
1. As expected, the correlation is consistently positive.
2. The R dataset cannot be taken seriously because a regression incident in a run
was automatically assigned MYCN_final=0 and MYCN_final_init=0.
3. The correlation is stronger in the O dataset than in the D dataset. This is
unsurprising as MYCN_final_init is always bigger than 1 in the O dataset but not
the D dataset.
"""
MYCN_init = numpy.array(df_processed.iloc[:,29])
MYCN_final = numpy.array(df_processed.iloc[:,30])
MYCN_final_init = numpy.array(df_processed.iloc[:,31])
plt.scatter(MYCN_init, MYCN_final)
plt.xlabel("MYCN_init")
plt.ylabel("MYCN_final")
plt.scatter(MYCN_init, MYCN_final_init)
plt.xlabel("MYCN_init")
plt.ylabel("MYCN_final_init")

MYCN_init_R = numpy.array(df_processed_R.iloc[:,29])
MYCN_final_R = numpy.array(df_processed_R.iloc[:,30])
MYCN_final_init_R = numpy.array(df_processed_R.iloc[:,31])
plt.scatter(MYCN_init_R, MYCN_final_R)
plt.xlabel("MYCN_init")
plt.ylabel("MYCN_final")
plt.scatter(MYCN_init_R, MYCN_final_init_R)
plt.xlabel("MYCN_init")
plt.ylabel("MYCN_final_init")

MYCN_init_D = numpy.array(df_processed_D.iloc[:,29])
MYCN_final_D = numpy.array(df_processed_D.iloc[:,30])
MYCN_final_init_D = numpy.array(df_processed_D.iloc[:,31])
plt.scatter(MYCN_init_D, MYCN_final_D)
plt.xlabel("MYCN_init")
plt.ylabel("MYCN_final")
plt.scatter(MYCN_init_D, MYCN_final_init_D)
plt.xlabel("MYCN_init")
plt.ylabel("MYCN_final_init")

MYCN_init_O = numpy.array(df_processed_O.iloc[:,29])
MYCN_final_O = numpy.array(df_processed_O.iloc[:,30])
MYCN_final_init_O = numpy.array(df_processed_O.iloc[:,31])
plt.scatter(MYCN_init_O, MYCN_final_O)
plt.xlabel("MYCN_init")
plt.ylabel("MYCN_final")
plt.scatter(MYCN_init_O, MYCN_final_init_O)
plt.xlabel("MYCN_init")
plt.ylabel("MYCN_final_init")

"""
Principal component analysis.
1. That the inputs in the original dataset can be mapped to a lower dimensional space means the 28 inputs are correlated, possibly due to noises and the fact that the 24 clonal fractions are not uniformly distributed.
2. The principal components derived for the R and O datasets are more predictive than those in the original and D datasets.
3. The results are consistent with the narrow range of O2 in the R dataset and the narrow range of theta_sc in the O dataset.
"""
pca=skldecomp.PCA(n_components=28, random_state=42)
pca.fit(df_processed.iloc[:,1:29])
plt.bar(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28'], pca.explained_variance_ratio_, align='center')
sum(pca.explained_variance_ratio_[0:2])

pca_R=skldecomp.PCA(n_components=28, random_state=42)
pca_R.fit(df_processed_R.iloc[:,1:29])
plt.bar(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28'], pca_R.explained_variance_ratio_, align='center')
sum(pca_R.explained_variance_ratio_[0:2])

pca_D=skldecomp.PCA(n_components=28, random_state=42)
pca_D.fit(df_processed_D.iloc[:,1:29])
plt.bar(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28'], pca_D.explained_variance_ratio_, align='center')
sum(pca_D.explained_variance_ratio_[0:2])

pca_O=skldecomp.PCA(n_components=28, random_state=42)
pca_O.fit(df_processed_O.iloc[:,1:29])
plt.bar(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28'], pca_O.explained_variance_ratio_, align='center')
sum(pca_O.explained_variance_ratio_[0:2])

"""
Merging of the R and O datasets.
"""
df_processed_RO = pandas.concat([df_processed_R,df_processed_O])
df_processed_RO=df_processed_RO.sort_values('Index')

"""
Exploratory analysis: the RO dataset.
1. It looks like there are two clusters: low O2 and low theta_sc.
2. Potentially one for R and one for O.
"""
plt.scatter(df_processed_RO.iloc[:,1], df_processed_RO.iloc[:,2])
plt.scatter(df_processed_RO.iloc[:,1], df_processed_RO.iloc[:,3])
plt.scatter(df_processed_RO.iloc[:,1], df_processed_RO.iloc[:,4])

"""
Clustering of the RO dataset: only O2 and theta_sc.
1. Hierarchical clustering.
2. Hierarchical clustering: adjusted rand index (0.565).
3. Hierarchical clustering: silhouette coefficients without (0.545) and with noises (0.526).
4. Hierarchical clustering: dendrogram without noises.
5. DBSCAN: hyperparametric sweep using the adjusted rand index.
6. DBSCAN: test the best set of clusters using the adjusted rand index (0.656).

By the silhouette coefficient without noises, this is the best. It supports the idea
that low O2 and low theta_sc are two distinct clusters, but it is unclear whether
they correspond to R and O.
"""
HC_RO_O2SC = sklcluster.AgglomerativeClustering(n_clusters=2, linkage='ward')
HC_RO_O2SC.fit_predict(df_processed_RO.iloc[:,[1,3]])

sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, HC_RO_O2SC.labels_)

sklmetrics.silhouette_score(df_processed_RO.iloc[:,[1,3]], HC_RO_O2SC.labels_)
sklmetrics.silhouette_score(df_processed_RO.iloc[:,[1,29]], HC_RO_O2SC.labels_)

Dendrogen_RO_O2SC = spcluster.ward(df_processed_RO.iloc[:,[1,3]])
spcluster.dendrogram(Dendrogen_RO_O2SC)

i_best = -1
j_best = -1
ARS_best = -1
for i in numpy.arange(0.01, 6, 0.01):
    for j in range(2, 130):
        DB_dummy = sklcluster.DBSCAN(eps=i, min_samples=j)
        DB_dummy.fit(df_processed_RO.iloc[:,[1,3]])
        if len(set(DB_dummy.labels_)) == 1:
            if -1 in set(DB_dummy.labels_):
                n_cluster = 0
            else:
                n_cluster = 1
        else:
            if -1 in set(DB_dummy.labels_):
                n_cluster = len(set(DB_dummy.labels_))-1
            else:
                n_cluster = len(set(DB_dummy.labels_))
        if n_cluster == 2:
            ARS_dummy = sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, DB_dummy.labels_)
            if ARS_dummy > ARS_best:
                print(ARS_dummy)
                ARS_best = ARS_dummy
                i_best = i
                j_best = j

DB_RO_O2SC = sklcluster.DBSCAN(eps=i_best, min_samples=j_best)
DB_RO_O2SC.fit(df_processed_RO.iloc[:,[1,3]])
sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, DB_RO_O2SC.labels_)

"""
Clustering of the RO dataset: only the macroscopic features.
1. Hierarchical clustering.
2. Hierarchical clustering: adjusted rand index (0.636).
3. Hierarchical clustering: silhouette coefficients without (0.309) and with noises (0.302).
4. Hierarchical clustering: dendrogram without noises.
5. DBSCAN: hyperparametric sweep using the adjusted rand index.
6. DBSCAN: test the best set of clusters using the adjusted rand index (0.581).

By the adjusted rand index for HC, this is the best. It suggests that cellularity
and degdiff are useful features too and we cannot just consider O2 and theta_sc.
"""
HC_RO_macro = sklcluster.AgglomerativeClustering(n_clusters=2, linkage='ward')
HC_RO_macro.fit_predict(df_processed_RO.iloc[:,1:5])

sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, HC_RO_macro.labels_)

sklmetrics.silhouette_score(df_processed_RO.iloc[:,[1,5]], HC_RO_macro.labels_)
sklmetrics.silhouette_score(df_processed_RO.iloc[:,[1,29]], HC_RO_macro.labels_)

Dendrogram_RO_macro = spcluster.ward(df_processed_RO.iloc[:,1:5])
spcluster.dendrogram(Dendrogram_RO_macro)

i_best = -1
j_best = -1
ARS_best = -1
for i in numpy.arange(0.01, 6, 0.01):
    for j in range(2, 130):
        DB_dummy = sklcluster.DBSCAN(eps=i, min_samples=j)
        DB_dummy.fit(df_processed_RO.iloc[:,[1,5]])
        if len(set(DB_dummy.labels_)) == 1:
            if -1 in set(DB_dummy.labels_):
                n_cluster = 0
            else:
                n_cluster = 1
        else:
            if -1 in set(DB_dummy.labels_):
                n_cluster = len(set(DB_dummy.labels_))-1
            else:
                n_cluster = len(set(DB_dummy.labels_))
        if n_cluster == 2:
            ARS_dummy = sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, DB_dummy.labels_)
            if ARS_dummy > ARS_best:
                print(ARS_dummy)
                ARS_best = ARS_dummy
                i_best = i
                j_best = j

DB_RO_macro = sklcluster.DBSCAN(eps=i_best, min_samples=j_best)
DB_RO_macro.fit(df_processed_RO.iloc[:,[1,5]])
sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, DB_RO_macro.labels_)

"""
Clustering of the RO dataset: all the features.
1. Hierarchical clustering.
2. Hierarchical clustering: adjusted rand index (0.542) and silhouette coefficient (0.405).
3. Hierarchical clustering: dendrogram.
4. DBSCAN: hyperparametric sweep using the adjusted rand index.
5. DBSCAN: test the best set of clusters using the adjusted rand index (0.670).

By the adjusted rand index for DBSCAN, this is the best. As DBSCAN filters out noises, it
supports the idea that the 24 microscopic features are noises.
"""
HC_RO_all = sklcluster.AgglomerativeClustering(n_clusters=2, linkage='ward')
HC_RO_all.fit_predict(df_processed_RO.iloc[:,1:29])

sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, HC_RO_all.labels_)
sklmetrics.silhouette_score(df_processed_RO.iloc[:,[1,29]], HC_RO_all.labels_)

Dendrogram_RO_all = spcluster.ward(df_processed_RO.iloc[:,1:29])
spcluster.dendrogram(Dendrogram_RO_all)

i_best = -1
j_best = -1
ARS_best = -1
for i in numpy.arange(0.01, 6, 0.01):
    for j in range(2, 130):
        DB_dummy = sklcluster.DBSCAN(eps=i, min_samples=j)
        DB_dummy.fit(df_processed_RO.iloc[:,[1,29]])
        if len(set(DB_dummy.labels_)) == 1:
            if -1 in set(DB_dummy.labels_):
                n_cluster = 0
            else:
                n_cluster = 1
        else:
            if -1 in set(DB_dummy.labels_):
                n_cluster = len(set(DB_dummy.labels_))-1
            else:
                n_cluster = len(set(DB_dummy.labels_))
        if n_cluster == 2:
            ARS_dummy = sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, DB_dummy.labels_)
            if ARS_dummy > ARS_best:
                print(ARS_dummy)
                ARS_best = ARS_dummy
                i_best = i
                j_best = j

DB_RO_all = sklcluster.DBSCAN(eps=i_best, min_samples=j_best)
DB_RO_all.fit(df_processed_RO.iloc[:,[1,29]])
sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, DB_RO_all.labels_)

"""
Principal component analysis of the RO dataset's macroscopic features.
1. The macroscopic features can be condensed into more abstract/general features.
"""
pca_RO=skldecomp.PCA(n_components=4, random_state=42)
pca_RO.fit(df_processed_RO.iloc[:,1:5])
plt.bar(['1', '2', '3', '4'], pca_RO.explained_variance_ratio_, align='center')

"""
Transform the RO dataset's macroscopic features.
"""
macro_pca = pca_RO.fit_transform(df_processed_RO.iloc[:,1:5])

"""
Clustering of the RO dataset, using all the transformed macroscopic features.
1. Hierarchical clustering.
2. Hierarchical clustering: adjusted rand index (0.636).
3. Hierarchical clustering: silhouette coefficients with different levels of noises (0.315 and 0.138).
4. Hierarchical clustering: dendrogram without noises.
5. DBSCAN: hyperparametric sweep using the adjusted rand index.
6. DBSCAN: test the best set of clusters using the adjusted rand index (0.456).
"""
HC_RO_macropca = sklcluster.AgglomerativeClustering(n_clusters=2, linkage='ward')
HC_RO_macropca.fit_predict(macro_pca)

sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, HC_RO_macropca.labels_)

sklmetrics.silhouette_score(macro_pca, HC_RO_macropca.labels_)
sklmetrics.silhouette_score(df_processed_RO.iloc[:,1:29], HC_RO_macropca.labels_)

Dendrogen_RO_macropca = spcluster.ward(macro_pca)
spcluster.dendrogram(Dendrogen_RO_macropca)

i_best = -1
j_best = -1
ARS_best = -1
for i in numpy.arange(0.01, 6, 0.01):
    for j in range(2, 130):
        DB_dummy = sklcluster.DBSCAN(eps=i, min_samples=j)
        DB_dummy.fit(macro_pca)
        if len(set(DB_dummy.labels_)) == 1:
            if -1 in set(DB_dummy.labels_):
                n_cluster = 0
            else:
                n_cluster = 1
        else:
            if -1 in set(DB_dummy.labels_):
                n_cluster = len(set(DB_dummy.labels_))-1
            else:
                n_cluster = len(set(DB_dummy.labels_))
        if n_cluster == 2:
            ARS_dummy = sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, DB_dummy.labels_)
            if ARS_dummy > ARS_best:
                print(ARS_dummy)
                ARS_best = ARS_dummy
                i_best = i
                j_best = j

DB_RO_macropca = sklcluster.DBSCAN(eps=i_best, min_samples=j_best)
DB_RO_macropca.fit(macro_pca)
sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, DB_RO_macropca.labels_)

"""
Clustering of the RO dataset, using the top 3 transformed macroscopic features.
1. Hierarchical clustering.
2. Hierarchical clustering: adjusted rand index (0.687).
3. Hierarchical clustering: silhouette coefficients with different levels of noises (0.335, 0.312, and 0.315).
4. Hierarchical clustering: dendrogram without noises.
5. DBSCAN: hyperparametric sweep using the adjusted rand index.
6. DBSCAN: test the best set of clusters using the adjusted rand index (0.284).
"""
HC_RO_macropca2 = sklcluster.AgglomerativeClustering(n_clusters=2, linkage='ward')
HC_RO_macropca2.fit_predict(macro_pca[:,0:3])

sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, HC_RO_macropca2.labels_)

sklmetrics.silhouette_score(macro_pca[:,0:3], HC_RO_macropca2.labels_)
sklmetrics.silhouette_score(macro_pca, HC_RO_macropca2.labels_)
sklmetrics.silhouette_score(df_processed_RO.iloc[:,1:29], HC_RO_macropca2.labels_)

Dendrogen_RO_macropca2 = spcluster.ward(macro_pca[:,0:3])
spcluster.dendrogram(Dendrogen_RO_macropca2)

i_best = -1
j_best = -1
ARS_best = -1
for i in numpy.arange(0.01, 6, 0.01):
    for j in range(2, 130):
        DB_dummy = sklcluster.DBSCAN(eps=i, min_samples=j)
        DB_dummy.fit(macro_pca[:,0:3])
        if len(set(DB_dummy.labels_)) == 1:
            if -1 in set(DB_dummy.labels_):
                n_cluster = 0
            else:
                n_cluster = 1
        else:
            if -1 in set(DB_dummy.labels_):
                n_cluster = len(set(DB_dummy.labels_))-1
            else:
                n_cluster = len(set(DB_dummy.labels_))
        if n_cluster == 2:
            ARS_dummy = sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, DB_dummy.labels_)
            if ARS_dummy > ARS_best:
                print(ARS_dummy)
                ARS_best = ARS_dummy
                i_best = i
                j_best = j

DB_RO_macropca2 = sklcluster.DBSCAN(eps=i_best, min_samples=j_best)
DB_RO_macropca2.fit(macro_pca[:,0:3])
sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, DB_RO_macropca2.labels_)

"""
Clustering of the RO dataset, using the top 2 transformed macroscopic features.
1. Hierarchical clustering.
2. Hierarchical clustering: adjusted rand index (0.766).
3. Hierarchical clustering: silhouette coefficients with different levels of noises (0.431, 0.325, and 0.142).
4. Hierarchical clustering: dendrogram without noises.
5. DBSCAN: hyperparametric sweep using the adjusted rand index.
6. DBSCAN: test the best set of clusters using the adjusted rand index (0.573).

By the adjusted rand index for HC, this is the best. It suggests that although
O2 and theta_sc might not be the only useful features, there might only be two
driving mechanisms represented by two composite features derived from the
four macroscopic features.
"""
HC_RO_macropca3 = sklcluster.AgglomerativeClustering(n_clusters=2, linkage='ward')
HC_RO_macropca3.fit_predict(macro_pca[:,0:2])

sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, HC_RO_macropca3.labels_)

sklmetrics.silhouette_score(macro_pca[:,0:2], HC_RO_macropca3.labels_)
sklmetrics.silhouette_score(macro_pca, HC_RO_macropca3.labels_)
sklmetrics.silhouette_score(df_processed_RO.iloc[:,1:29], HC_RO_macropca3.labels_)

Dendrogen_RO_macropca3 = spcluster.ward(macro_pca[:,0:2])
spcluster.dendrogram(Dendrogen_RO_macropca3)

i_best = -1
j_best = -1
ARS_best = -1
for i in numpy.arange(0.01, 6, 0.01):
    for j in range(2, 130):
        DB_dummy = sklcluster.DBSCAN(eps=i, min_samples=j)
        DB_dummy.fit(macro_pca[:,0:2])
        if len(set(DB_dummy.labels_)) == 1:
            if -1 in set(DB_dummy.labels_):
                n_cluster = 0
            else:
                n_cluster = 1
        else:
            if -1 in set(DB_dummy.labels_):
                n_cluster = len(set(DB_dummy.labels_))-1
            else:
                n_cluster = len(set(DB_dummy.labels_))
        if n_cluster == 2:
            ARS_dummy = sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, DB_dummy.labels_)
            if ARS_dummy > ARS_best:
                print(ARS_dummy)
                ARS_best = ARS_dummy
                i_best = i
                j_best = j

DB_RO_macropca3 = sklcluster.DBSCAN(eps=i_best, min_samples=j_best)
DB_RO_macropca3.fit(macro_pca[:,0:2])
sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, DB_RO_macropca3.labels_)

"""
Clustering of the RO dataset, using the top transformed macroscopic feature.
1. Hierarchical clustering.
2. Hierarchical clustering: adjusted rand index (0.657).
3. Hierarchical clustering: silhouette coefficients with different levels of noises (0.557, 0.320, and 0.145).
4. Hierarchical clustering: dendrogram without noises.
5. DBSCAN: hyperparametric sweep using the adjusted rand index.
6. DBSCAN: test the best set of clusters using the adjusted rand index (0.748).

By the silhouette coefficient without noises and the adjusted rand index for DBSCAN,
this is the best. The first one is unsurprising as the principal component describes
most of the variations in the data, so the data points must be well-separated along
this component; including the secondary components will just draw them closer together.
The second one has a similar explanation. Along the principal component, it is the easiest
to identify outliers.
"""
HC_RO_macropca4 = sklcluster.AgglomerativeClustering(n_clusters=2, linkage='ward')
HC_RO_macropca4.fit_predict(macro_pca[:,0:1])

sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, HC_RO_macropca4.labels_)

sklmetrics.silhouette_score(macro_pca[:,0:1], HC_RO_macropca4.labels_)
sklmetrics.silhouette_score(macro_pca, HC_RO_macropca4.labels_)
sklmetrics.silhouette_score(df_processed_RO.iloc[:,1:29], HC_RO_macropca4.labels_)

Dendrogen_RO_macropca4 = spcluster.ward(macro_pca[:,0:1])
spcluster.dendrogram(Dendrogen_RO_macropca4)

i_best = -1
j_best = -1
ARS_best = -1
for i in numpy.arange(0.01, 6, 0.01):
    for j in range(2, 130):
        DB_dummy = sklcluster.DBSCAN(eps=i, min_samples=j)
        DB_dummy.fit(macro_pca[:,0:1])
        if len(set(DB_dummy.labels_)) == 1:
            if -1 in set(DB_dummy.labels_):
                n_cluster = 0
            else:
                n_cluster = 1
        else:
            if -1 in set(DB_dummy.labels_):
                n_cluster = len(set(DB_dummy.labels_))-1
            else:
                n_cluster = len(set(DB_dummy.labels_))
        if n_cluster == 2:
            ARS_dummy = sklmetrics.adjusted_rand_score(df_processed_RO.Lablel, DB_dummy.labels_)
            if ARS_dummy > ARS_best:
                print(ARS_dummy)
                ARS_best = ARS_dummy
                i_best = i
                j_best = j

"""
Further investigations into the R dataset.
1. Hierarchical clustering: just the macroscopic variables, informed by the above analysis.
2. Hierarchical clustering: silhouette coefficient without noises (0.522).
3. Hierarchical clustering: dendrogram without noises.
4. Hierarchical clustering: summary statistics of the two clusters in the R dataset.
(a) O2 is low in both.
(b) One has high cellularity, low theta_sc, and low degdiff.
(c) One has low cellularity, high theta_sc, and high degdiff.
5. Principal component analysis of the macroscopic variables only.
(a) There are three latent features defining the variations inside this dataset.
(b) This is expected as O2 is confined to a low value.
"""
HC_R_macro = sklcluster.AgglomerativeClustering(n_clusters=2, linkage='ward')
HC_R_macro.fit_predict(df_processed_R.iloc[:,[1,5]])

sklmetrics.silhouette_score(df_processed_R.iloc[:,[1,5]], HC_R_macro.labels_)

Dendrogen_R_macro = spcluster.ward(df_processed_R.iloc[:,[1,5]])
spcluster.dendrogram(Dendrogen_R_macro)

df_processed_R['SC1']=HC_R_macro.labels_
df_processed_R0 = df_processed_R[df_processed_R['SC1']==0]
df_processed_R1 = df_processed_R[df_processed_R['SC1']==1]
df_processed_R0.iloc[:,1:5].describe()
df_processed_R1.iloc[:,1:5].describe()

pca_Rmacro=skldecomp.PCA(n_components=4, random_state=42)
pca_Rmacro.fit(df_processed_R.iloc[:,1:5])
plt.bar(['1', '2', '3', '4'], pca_Rmacro.explained_variance_ratio_, align='center')

"""
Further investigations into the O dataset.
1. Hierarchical clustering: just the macroscopic variables, informed by the above analysis.
2. Hierarchical clustering: silhouette coefficient without noises (0.444).
3. Hierarchical clustering: dendrogram without noises.
4. Hierarchical clustering: summary statistics of the two clusters in the O dataset.
(a) theta_sc is low in both.
(b) One has low O2 and high degdiff. This is interesting. Differentiated cells do not proliferate as much, so they do not need as much oxygen. As a result, the relatively low O2 level cannot force them into regression.
(c) One has high O2 and low degdiff.
(d) Cellularity does not differ much between the two clusters.
5. Principal component analysis of the macroscopic variables only.
(a) There are three latent features  defining the variations inside this dataset.
(b) This is expected as theta_sc is confined to a low value.
"""
HC_O_macro = sklcluster.AgglomerativeClustering(n_clusters=2, linkage='ward')
HC_O_macro.fit_predict(df_processed_O.iloc[:,[1,5]])

sklmetrics.silhouette_score(df_processed_O.iloc[:,[1,5]], HC_O_macro.labels_)

Dendrogen_O_macro = spcluster.ward(df_processed_O.iloc[:,[1,5]])
spcluster.dendrogram(Dendrogen_O_macro)

df_processed_O['SC1']=HC_O_macro.labels_
df_processed_O0 = df_processed_O[df_processed_O['SC1']==0]
df_processed_O1 = df_processed_O[df_processed_O['SC1']==1]
df_processed_O0.iloc[:,1:5].describe()
df_processed_O1.iloc[:,1:5].describe()

pca_Omacro=skldecomp.PCA(n_components=4, random_state=42)
pca_Omacro.fit(df_processed_O.iloc[:,1:5])
plt.bar(['1', '2', '3', '4'], pca_Omacro.explained_variance_ratio_, align='center')