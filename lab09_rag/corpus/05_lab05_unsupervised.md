# Lab 05 — Unsupervised Learning: Clustering and Dimensionality Reduction

Lab 05 segments mobile-money customers without using the fraud label at all.

## K-Means

K-Means partitions data into k clusters by alternating two steps: assign each point to the
nearest centroid, then move each centroid to the mean of its assigned points. It minimises
within-cluster sum of squares (inertia).

K-Means assumes clusters are roughly spherical and of similar size, and it is sensitive to
feature scale — so features must be standardised first. It is also sensitive to
initialisation, which is why `k-means++` initialisation and `n_init=10` are used.

## Choosing k

Two diagnostics are used. The **elbow method** plots inertia against k and looks for the
bend where additional clusters stop buying much reduction. The **silhouette score**
measures how close each point is to its own cluster compared with the nearest other
cluster, on a scale from -1 to 1; higher is better. On this dataset the silhouette peaks at
k = 4.

## DBSCAN

DBSCAN groups points by density rather than distance to a centroid. It takes `eps` (the
neighbourhood radius) and `min_samples`. Unlike K-Means it finds clusters of arbitrary
shape, does not require k to be specified in advance, and explicitly labels outliers as
noise with the label -1 — which makes it a natural fit for anomaly detection.

## PCA

Principal Component Analysis projects the data onto the orthogonal directions of greatest
variance. The lab keeps enough components to explain 95% of the variance, which on this
feature set takes 11 of the original 24 components. PCA is a linear method and is fitted on
the training set only, then applied to the test set.

## t-SNE and UMAP

t-SNE and UMAP are non-linear methods used **for visualisation only**. Their axes are not
interpretable, distances between well-separated clusters are not meaningful, and neither
should be used as a preprocessing step feeding a classifier.

## The headline caution

Clusters are not labels. A cluster with a high fraud rate is a hypothesis to investigate,
not a fraud detector. The lab ends by cross-tabulating cluster assignment against the
withheld `is_fraud` flag purely as a sanity check.
