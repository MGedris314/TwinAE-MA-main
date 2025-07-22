import numpy as np
import matplotlib.pyplot as plt


def cluster_data(data, method='kmeans', **kwargs):
    """
    Cluster the input data using the specified clustering method.

    Parameters:
    - data: array-like, shape (n_samples, n_features)
        The input data to cluster.
    - method: str, optional (default='kmeans')
        The clustering method to use. Options are 'kmeans', 'dbscan', 'agglomerative', 'meanshift', 'spectral'.
    - kwargs: additional keyword arguments for the clustering algorithm.

    Returns:
    - labels: array, shape (n_samples,)
        The labels of each point in the dataset.
    """
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, SpectralClustering

    
    if method == 'kmeans':
        model = KMeans(**kwargs)
    elif method == 'dbscan':
        model = DBSCAN(**kwargs)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(**kwargs)
    elif method == 'meanshift':
        model = MeanShift(**kwargs)
    elif method == 'spectral':
        model = SpectralClustering(**kwargs)
    else:
        raise ValueError("Unknown clustering method: {}".format(method))

    labels = model.fit_predict(data)
    return labels

def plot_clusters(data, method = "kmeans", title='Cluster Plot', cmap='tab10', alpha=0.8, figsize=(10, 6), **kwargs):
    """
    Plot the clusters of the data using a scatter plot.

    Parameters:
    - data: array-like, shape (n_samples, n_features)
        The input data to visualize.
    - labels: array-like, shape (n_samples,)
        The cluster labels for each point in the dataset.
    - title: str, optional
        Title for the plot.
    - cmap: str or Colormap, optional
        Colormap for the scatter plot.
    - alpha: float, optional
        Transparency for the points.
    - figsize: tuple, optional
        Figure size.
    """
    labels = cluster_data(data, method=method, **kwargs)

    plt.figure(figsize=figsize)
    sc = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, alpha=alpha)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(sc)
    plt.show()

def pca_scatter_plot(data_a, labels_a, data_b = None, labels_b = None,
                                 title_a='PCA Plot A', title_b='PCA Plot B',
                                 cmap='tab10', alpha=0.8, figsize=(16, 6)):
    """
    Plot two PCA scatter plots of the same data side by side, using two different label sets.

    Parameters:
    - data: array-like, shape (n_samples, n_features)
        The input data to visualize.
    - labels_a: array-like, shape (n_samples,)
        The first set of cluster labels.
    - labels_b: array-like, shape (n_samples,)
        The second set of cluster labels.
    - title_a: str, optional
        Title for the first subplot.
    - title_b: str, optional
        Title for the second subplot.
    - cmap: str or Colormap, optional
        Colormap for the scatter plots.
    - alpha: float, optional
        Transparency for the points.
    - figsize: tuple, optional
        Figure size.
    """
    from sklearn.decomposition import PCA

    # reduce once and share coordinates
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data_a)

    if data_b is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        sc0 = axes[0].scatter(reduced[:, 0], reduced[:, 1],
                            c=labels_a, cmap=cmap, alpha=alpha)
        axes[0].set_title(title_a)
        axes[0].set_xlabel('PCA Component 1')
        axes[0].set_ylabel('PCA Component 2')
        plt.colorbar(sc0, ax=axes[0])
    else:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        sc0 = axes.scatter(reduced[:, 0], reduced[:, 1],
                            c=labels_a, cmap=cmap, alpha=alpha)
        axes.set_title(title_a)
        axes.set_xlabel('PCA Component 1')
        axes.set_ylabel('PCA Component 2')
        plt.colorbar(sc0, ax=axes)

    if data_b is not None:
        # reduce once and share coordinates
        reduced = pca.fit_transform(data_b)

        sc1 = axes[1].scatter(reduced[:, 0], reduced[:, 1],
                                c=labels_b, cmap=cmap, alpha=alpha)
        axes[1].set_title(title_b)
        axes[1].set_xlabel('PCA Component 1')
        axes[1].set_ylabel('PCA Component 2')
        plt.colorbar(sc1, ax=axes[1])

        plt.tight_layout()
        plt.show()

def create_anchors(dataset_size):
    """Returns an array of anchors equal to the datset size."""
    import random
    random.seed(42)
    rand_ints = random.sample(range(dataset_size), dataset_size)
    return np.vstack([rand_ints, rand_ints]).T

def find_low_fidelity_points(data, anchors, clusters_to_run = [3,4,5,6,7,8], plot = True, verbose = True):
    """
    anchor : this should be a list of indices of the data points that we want to use as anchors
    # Lets Identify points that often are belonging to different clusters
        - We can keep track of how many times each point is in a cluster with each anchor
        - Then we can count how many unique combinations they have
        - Shape of matrix would be [num-points, num-clustering algorithms, num_anchors]
    """

    fidelity_matrix = np.zeros((data.shape[0], len(clusters_to_run), len(anchors)))

    #Repeate for each number of clusters
    for n_clusters in clusters_to_run:

        #Get the labels
        labels = cluster_data(data, method='kmeans', n_clusters=n_clusters)
        
        # For each point, determine if each anchor is in the same cluster
        for anchor_idx, anchor in enumerate(anchors):
            for point_idx, point in enumerate(data):
                if labels[point_idx] == labels[anchor]:
                    fidelity_matrix[point_idx, clusters_to_run.index(n_clusters), anchor_idx] = 1
                else:
                    fidelity_matrix[point_idx, clusters_to_run.index(n_clusters), anchor_idx] = 0

    # Count unique combinations of clusters for each point
    unique_combinations = np.array([
                            len(np.unique(fidelity_matrix[i], axis=0)) 
                            for i in range(data.shape[0])
                        ])
    
    # Identify low fidelity points (points with the most unique combinations)
    low_fidelity_points = np.argsort(unique_combinations)[-10:]  #? Currently finds the points with the most unique combinations

    if verbose:
        # Print the actual data points rather than just their indices
        print("Low fidelity points data:")
        print(data[low_fidelity_points])
        print('-'*60, f"\nUnique combinations per point: {unique_combinations}")
        print('-'*60, f"\nFidelity Matrix: {fidelity_matrix}")


    if plot:
        # Highlight low-fidelity points on the original data scatter
        mask = np.zeros(data.shape[0], dtype=bool)
        mask[low_fidelity_points] = True

        plt.figure(figsize=(10, 6))
        # plot high-fidelity points
        plt.scatter(
            data[~mask, 0], data[~mask, 1],
            c='blue', label='High Fidelity', alpha=0.6
        )
        # plot low-fidelity points
        plt.scatter(
            data[mask, 0], data[mask, 1],
            c='red', label='Low Fidelity', alpha=0.8
        )
        plt.title('Data with Low vs High Fidelity Points')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()


    return low_fidelity_points

def adjust_anchors(anchors, points_removed_A, points_removed_B):
    """
    Adjusts the anchors based on the points removed from two datasets.

    Parameters:
    - anchors: array-like, shape (n_anchors, 2)
        The original anchors where each row is [index_in_A, index_in_B].
    - points_removed_A: array-like, shape (n_removed_A,)
        Indices of points removed from dataset A.
    - points_removed_B: array-like, shape (n_removed_B,)
        Indices of points removed from dataset B.

    Returns:
    - adjusted_anchors: array-like, shape (n_anchors, 2)
        The adjusted anchors after removing the specified points.
        Each anchor index is adjusted to account for removed points.
    """
    # Convert to numpy arrays for easier manipulation
    anchors = np.array(anchors)
    points_removed_A = np.array(points_removed_A) if len(points_removed_A) > 0 else np.array([])
    points_removed_B = np.array(points_removed_B) if len(points_removed_B) > 0 else np.array([])
    
    # Sort the removed points in descending order to avoid index shifting issues
    points_removed_A = np.sort(points_removed_A)
    points_removed_B = np.sort(points_removed_B)
    
    adjusted_anchors = anchors.copy()
    
    # Adjust anchors for dataset A (first column)
    for anchor_idx in range(len(adjusted_anchors)):
        original_idx_A = anchors[anchor_idx, 0]
        # Count how many removed points have indices less than the current anchor
        num_removed_before = np.sum(points_removed_A < original_idx_A)
        adjusted_anchors[anchor_idx, 0] = original_idx_A - num_removed_before
    
    # Adjust anchors for dataset B (second column)
    for anchor_idx in range(len(adjusted_anchors)):
        original_idx_B = anchors[anchor_idx, 1]
        # Count how many removed points have indices less than the current anchor
        num_removed_before = np.sum(points_removed_B < original_idx_B)
        adjusted_anchors[anchor_idx, 1] = original_idx_B - num_removed_before
    
    return adjusted_anchors

def remove_from_data(data, points_to_remove):
    """
    Removes specified points from the dataset.

    Parameters:
    - data: array-like, shape (n_samples, n_features)
        The input data from which points will be removed.
    - points_to_remove: array-like, shape (n_removed,)
        Indices of points to remove from the dataset.

    Returns:
    - adjusted_data: array-like, shape (n_samples - n_removed, n_features)
        The dataset after removing the specified points.
    """
    # Convert to numpy array for easier manipulation
    data = np.array(data)
    
    # Create a mask for the points to keep
    mask = np.ones(data.shape[0], dtype=bool)
    mask[points_to_remove] = False
    
    # Return the adjusted data
    return data[mask]