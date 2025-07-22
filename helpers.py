# Plot Embeddings helper Function
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import Categorical
from sklearn.decomposition import PCA
import pandas as pd

#Function that plots embeddings colored by label with lines connecting anchors
def plot_embeddings(emb, labels, anchors):
    #We are assuming the embeddings are half and half split (1 to 1 correspondence)
    styles = ['Domain A' if i < len(labels)/2 else 'Domain B' for i in range(len(emb))]

    plt.figure(figsize=(14, 8))
    ax = sns.scatterplot(x = emb[:, 0], y = emb[:, 1], style = styles, hue = Categorical(labels), s=120, markers= {"Domain A": "^", "Domain B" : "o"})

    #adjust anchors to match the reflected indicies
    known_anchors_adjusted = [(i, int(j + len(labels)/2)) for i, j in anchors]

    #Show lines between anchors
    for i in known_anchors_adjusted:
        ax.plot([emb[i[0], 0], emb[i[1], 0]], [emb[i[0], 1], emb[i[1], 1]], color = 'grey')
            
    plt.show()

def plot_domains(x=None, y=None, labels=None, title="Scatter Plot", colormap="viridis", emb=None, domain=None, data=None):
    # Ensure labels is a numpy array
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    # Create a colormap with as many colors as needed
    colors = plt.cm.get_cmap(colormap, len(unique_labels))
    
    plt.figure(figsize=(8,6))

    # PCA everything to be 2d
    if emb is not None and emb.shape[1] > 2:
        pca = PCA(n_components=2)
        emb = pca.fit_transform(emb)
    if domain is not None and domain.shape[1] > 2:
        pca = PCA(n_components=2)
        domain = pca.fit_transform(domain)
    if data is not None:
        if data.shape[1] > 2:
            pca = PCA(n_components=2)
            data = pca.fit_transform(data)
        x, y = data[:, 0], data[:, 1]

    if type(x) != None and type(y) != None:
        for i, label in enumerate(unique_labels):
            idx = labels == label
            plt.scatter(x[idx], y[idx], label=str(label), color=colors(i), s=70, alpha=0.8, edgecolor='black')
    
    if emb is not None:
        # Plot emb points as triangles; assuming emb is a (n,2) array
        plt.scatter(emb[:,0], emb[:,1], label="emb", marker="^", c=np.hstack([labels, labels]), s=40, alpha=0.8, edgecolor='black')

    if domain is not None:
        # Plot domain points as diamonds; assuming domain is a (n,2) array
        plt.scatter(domain[:,0], domain[:,1], label="Domain", marker="d", c=labels, s=40, alpha=0.8, edgecolor='black')

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

def prepare_dataset(filepath):
    """
    Takes a filepath to one of the provided csv files (they are formatted in a specific way). Will return
    the labels and the data.
    """

    #Read in file
    data = pd.read_csv(filepath).to_numpy()

    #Transform labels to numeric data
    labels = pd.Categorical(data[: , 0]).codes

    #Double labels (for plotting later)
    labels = np.concatenate([labels, labels])

    features = data[:, 1:].astype(np.float32)

    return features, labels

def split_features(features, split = "distort"):
        """
        Split the features to create distinct domains.

        Try setting split to "distort", "random", or "rotation". 

        See more here: PAPER_DESCRIPTION
        """

        import random
        random.seed(42)

        if split == "random":

            # Generate column indices and shuffle them
            column_indices = np.arange(features.shape[1])
            np.random.shuffle(column_indices)

            # Choose a random index to split the shuffled column indices
            split_index = random.randint(1, len(column_indices) - 1)

            # Use the shuffled indices to split the features array into two parts
            split_a = features[:, column_indices[:split_index]]
            split_b = features[:, column_indices[split_index:]]

        elif split == "rotation":
            #Apply random rotation to q
            rng = np.random.default_rng(42)
            d = np.shape(features)[1]
            random_matrix = rng.random((d, d))
            q, _ = np.linalg.qr(random_matrix)

            split_a = features

            #Transform features by q
            split_b = features @ q
        
        elif split == "distort":
            #Split A remains the same
            split_a = features

            #Add noise to split B
            split_b = features + np.random.normal(scale = 0.05, size = np.shape(features))

        #Reshape if they only have one sample
        if split_a.shape[1] == 1:
            split_a = split_a.reshape(-1, 1)
        if split_b.shape[1] == 1:
            split_b = split_b.reshape(-1, 1)


        return split_a, split_b

def create_anchors(dataset_size):
    """Returns an array of anchors equal to the datset size."""

    import random
    # print(range(dataset_size))
    # print(dataset_size)

    random.seed(42)

    #Generate anchors that can be subsetted
    rand_ints = random.sample(range(dataset_size), dataset_size)
    # print(np.vstack([rand_ints,rand_ints]).T)
    return np.vstack([rand_ints, rand_ints]).T