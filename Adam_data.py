"""
This file contains the classes to create the toy data for the experiments.

#? Whats with all the comments like #? or #* : Its the extension called "better comments" for how I highlight and organize my code.

Description of each class:
- ToyData: Basic class for inheritance
- S_Roll: S-Roll data set, with missing sections
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import plotly.express as px

# Add the parent directory to the path so we can import from Materials
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
# from Materials.personal_paths import BASEDIR
# from Materials.data_constants import SEEDS

SEEDS = [42,16272,36,832,150]

#& Basic Class for Inheritance
class ToyData():
    """
    This class has the underlying attirbutes for each of the toy data sets.
    """
    def __init__(self, n_samples, verbose = 1, seed = SEEDS[0]):
        self.n_samples = n_samples
        self.verbose = verbose
        self.seed = seed
        np.random.seed(self.seed)
        
        self.domainA, self.labelsA = self.generate_domainA()
        self.domainB, self.labelsB = self.generate_domainB()
    
    def generate_domainA(self):
        """
        This function generates the domain A data.
        """
        #! This is just a placeholder for the domain A data.
        raise NotImplementedError("This function is not implemented yet.")
    
    def generate_domainB(self):
        """
        This function generates the domain B data.
        """
        raise NotImplementedError("This function is not implemented yet.")

    def _prepare_data(self, data):
        n_dim = data.shape[1]
        if n_dim == 2:
            return data, '2d'
        elif n_dim == 3:
            return data, '3d'
        else:
            # PCA to 3D
            pca = PCA(n_components=3)
            data_pca = pca.fit_transform(data)
            return data_pca, '3d'

    def plot(self, **kwargs):
        """
        This function plots both domains of the data in two separate plots side by side using Matplotlib (2D or 3D as appropriate).
        """
        dataA, modeA = self._prepare_data(self.domainA)
        dataB, modeB = self._prepare_data(self.domainB)

        fig = plt.figure(figsize=(12, 5))

        # Plot Domain A
        if modeA == '3d':
            axA = fig.add_subplot(1, 2, 1, projection='3d')
            scatterA = axA.scatter(dataA[:, 0], dataA[:, 1], dataA[:, 2], c=self.labelsA, **kwargs)
            axA.set_xlabel('X1')
            axA.set_ylabel('X2')
            axA.set_zlabel('X3')
        else:
            axA = fig.add_subplot(1, 2, 1)
            scatterA = axA.scatter(dataA[:, 0], dataA[:, 1], c=self.labelsA, **kwargs)
            axA.set_xlabel('X1')
            axA.set_ylabel('X2')
        axA.set_title('Domain A')

        # Plot Domain B
        if modeB == '3d':
            axB = fig.add_subplot(1, 2, 2, projection='3d')
            scatterB = axB.scatter(dataB[:, 0], dataB[:, 1], dataB[:, 2], c=self.labelsB, **kwargs)
            axB.set_xlabel('X1')
            axB.set_ylabel('X2')
            axB.set_zlabel('X3')
        else:
            axB = fig.add_subplot(1, 2, 2)
            scatterB = axB.scatter(dataB[:, 0], dataB[:, 1], c=self.labelsB, **kwargs)
            axB.set_xlabel('X1')
            axB.set_ylabel('X2')
        axB.set_title('Domain B')

        plt.tight_layout()
        plt.show()

    def plot_together(self, **kwargs):
        """
        Plots both domains of the data in two or three dimensions on the same plot, or applies PCA to 3D if needed, using Plotly for interactivity.
        Ensures that the colors for each label/class are consistent between Domain A and Domain B.
        """
        dataA, modeA = self._prepare_data(self.domainA)
        dataB, modeB = self._prepare_data(self.domainB)
        mode = '3d' if (modeA == '3d' or modeB == '3d') else '2d'

        # Combine labels to get all unique classes
        all_labels = np.concatenate([self.labelsA, self.labelsB])
        unique_labels = np.unique(all_labels)
        n_classes = len(unique_labels)
        # Map each label to an index for color assignment
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        colorsA = np.array([label_to_index[l] for l in self.labelsA])
        colorsB = np.array([label_to_index[l] for l in self.labelsB])

        fig_width = 850
        fig_height = 440

        layout_margins = dict(l=7, r=7, t=30, b=7)  # reduce margins for less white border

        if mode == '3d':
            if dataA.shape[1] == 2:
                dataA = np.column_stack([dataA, np.zeros(dataA.shape[0])])
            if dataB.shape[1] == 2:
                dataB = np.column_stack([dataB, np.zeros(dataB.shape[0])])

            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=dataA[:, 0], y=dataA[:, 1], z=dataA[:, 2],
                mode='markers',
                marker=dict(size=4, color=colorsA, cmin=0, cmax=n_classes-1),
                name='Domain A',
                **kwargs
            ))
            fig.add_trace(go.Scatter3d(
                x=dataB[:, 0], y=dataB[:, 1], z=dataB[:, 2],
                mode='markers',
                marker=dict(size=4, color=colorsB, cmin=0, cmax=n_classes-1),
                name='Domain B',
                **kwargs
            ))
            fig.update_layout(
                scene=dict(
                    xaxis_title='X1', yaxis_title='X2', zaxis_title='X3'
                ),
                title='Domains A and B Together (Plotly 3D)',
                width=fig_width,
                height=fig_height,
                margin=layout_margins
            )
            # Add a colorbar for class labels
            fig.update_traces(marker_colorbar=dict(title='Class', tickvals=list(range(n_classes)), ticktext=[str(l) for l in unique_labels]))
            fig.show()
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dataA[:, 0], y=dataA[:, 1],
                mode='markers',
                marker=dict(size=6, color=colorsA, cmin=0, cmax=n_classes-1),
                name='Domain A',
                **kwargs
            ))
            fig.add_trace(go.Scatter(
                x=dataB[:, 0], y=dataB[:, 1],
                mode='markers',
                marker=dict(size=6, color=colorsB, cmin=0, cmax=n_classes-1),
                name='Domain B',
                **kwargs
            ))
            fig.update_layout(
                xaxis_title='X1',
                yaxis_title='X2',
                title='Domains A and B Together (Plotly 2D)',
                width=fig_width,
                height=fig_height,
                margin=layout_margins
            )
            fig.show()

#& S-Roll and S-Curve
class S_Roll(ToyData):
    """
    This class generates the S-Roll data.
    Domain A: S-curve
    Domain B: Swirl/spiral
    """
    
    def __init__(self, n_samples, n_classes = 3, verbose=1, seed=SEEDS[0], delete_section=True):
        if verbose > 0:
            print(f"S_Roll.__init__ called with n_samples={n_samples}, n_classes={n_classes}, verbose={verbose}, seed={seed}, delete_section={delete_section}")
        
        self.n_classes = n_classes
        self.delete_section = delete_section
        super().__init__(n_samples, verbose, seed)
        
        if delete_section:
            self.remove_section('A')
        
        if verbose > 0:
            print(f"S_Roll.__init__ completed")
    
    def generate_domainA(self):
        """
        This function generates the domain A data (S-shape).
        The S shape is a classic 2D S-curve, often used in toy datasets.
        """
        if self.verbose > 0:
            print(f"generate_domainA called with n_samples={self.n_samples}, n_classes={self.n_classes}")

        from sklearn.datasets import make_s_curve
        
        # Generate more points than needed to ensure even distribution
        extra_samples = self.n_samples * 2
        data, t = make_s_curve(extra_samples, noise=0.02, random_state=self.seed)
        
        # Only use the first two dimensions (x, y) for 2D S-curve
        data = data[:, [0, 2]]

        # Create even-sized classes
        points_per_class = self.n_samples // self.n_classes
        
        # Normalize t to [0, 1] and create initial labels
        normalized_t = (t - t.min()) / (t.max() - t.min())
        initial_labels = np.floor(normalized_t * self.n_classes).astype(int)
        initial_labels = np.minimum(initial_labels, self.n_classes - 1)  # Ensure no overflow
        
        # Select points evenly from each class
        final_data = []
        final_labels = []
        
        for class_idx in range(self.n_classes):
            class_points = data[initial_labels == class_idx]
            if len(class_points) >= points_per_class:
                # Randomly select exactly points_per_class points
                selected_indices = np.random.choice(
                    len(class_points), 
                    points_per_class, 
                    replace=False
                )
                final_data.append(class_points[selected_indices])
                final_labels.extend([class_idx] * points_per_class)
            else:
                # If not enough points, take all and randomly duplicate
                n_duplicates = points_per_class - len(class_points)
                duplicated_indices = np.random.choice(
                    len(class_points), 
                    n_duplicates, 
                    replace=True
                )
                final_data.append(class_points)
                final_data.append(class_points[duplicated_indices])
                final_labels.extend([class_idx] * points_per_class)
        
        # Combine and shuffle
        final_data = np.vstack(final_data)
        final_labels = np.array(final_labels)
        
        if self.verbose > 0:
            print(f"generate_domainA completed - generated {len(final_data)} points")
            for i in range(self.n_classes):
                print(f"  Class {i}: {np.sum(final_labels == i)} points")
        
        return final_data, final_labels
    
    def generate_domainB(self):
        """
        This function generates the domain B data (tight swirl/spiral).
        Creates a more compact spiral with multiple rotations and even class distribution.
        """
        if self.verbose > 0:
            print(f"generate_domainB called with n_samples={self.n_samples}, n_classes={self.n_classes}")
        
        # Generate more points than needed to ensure even distribution
        extra_samples = self.n_samples * 2
        
        # Generate spiral data with more rotations and tighter spacing
        t = np.linspace(0, 6*np.pi, extra_samples)  # More rotations
        r = 0.1 + 0.4 * t / (6*np.pi)  # Start smaller, grow slower
        x = r * np.cos(t)
        y = r * np.sin(t)
        
        # Add some controlled noise
        noise = np.random.normal(0, 0.02, (extra_samples, 2))  # Reduced noise
        data = np.column_stack([x, y]) + noise
        
        # Create even-sized classes
        points_per_class = self.n_samples // self.n_classes
        
        # Calculate radial distance for initial labeling
        radial_dist = np.sqrt(x**2 + y**2)
        normalized_dist = (radial_dist - radial_dist.min()) / (radial_dist.max() - radial_dist.min())
        initial_labels = np.floor(normalized_dist * self.n_classes).astype(int)
        initial_labels = np.minimum(initial_labels, self.n_classes - 1)  # Ensure no overflow
        
        # Select points evenly from each class
        final_data = []
        final_labels = []
        
        for class_idx in range(self.n_classes):
            class_points = data[initial_labels == class_idx]
            if len(class_points) >= points_per_class:
                # Randomly select exactly points_per_class points
                selected_indices = np.random.choice(
                    len(class_points), 
                    points_per_class, 
                    replace=False
                )
                final_data.append(class_points[selected_indices])
                final_labels.extend([class_idx] * points_per_class)
            else:
                # If not enough points, take all and randomly duplicate
                n_duplicates = points_per_class - len(class_points)
                duplicated_indices = np.random.choice(
                    len(class_points), 
                    n_duplicates, 
                    replace=True
                )
                final_data.append(class_points)
                final_data.append(class_points[duplicated_indices])
                final_labels.extend([class_idx] * points_per_class)
        
        # Combine and shuffle
        final_data = np.vstack(final_data)
        final_labels = np.array(final_labels)
        
        if self.verbose > 0:
            print(f"generate_domainB completed - generated {len(final_data)} points")
            for i in range(self.n_classes):
                print(f"  Class {i}: {np.sum(final_labels == i)} points")
        
        return final_data, final_labels
    
    def remove_section(self, domain, start_frac=0.3, end_frac=0.7):
        """
        Remove a section of points from the specified domain.
        
        Parameters
        ----------
        domain : str
            Either 'A' or 'B' to specify which domain to modify
        start_frac : float
            Start fraction of the data to remove (0 to 1). This find the index of the point to remove.
        end_frac : float
            End fraction of the data to remove (0 to 1). This find the index of the point to remove.
        """
        if self.verbose > 0:
            print(f"remove_section called with domain={domain}, start_frac={start_frac}, end_frac={end_frac}")
        
        if domain == 'A':
            data = self.domainA
            labels = self.labelsA
        elif domain == 'B':
            data = self.domainB
            labels = self.labelsB
        else:
            raise ValueError("domain must be either 'A' or 'B'")
        
        # Calculate indices to remove
        n_points = len(data)
        start_idx = int(start_frac * n_points)
        end_idx = int(end_frac * n_points)
        
        # Create mask to keep points outside the section
        mask = np.ones(n_points, dtype=bool)
        mask[start_idx:end_idx] = False
        
        # Apply mask
        if domain == 'A':
            self.domainA = data[mask]
            self.labelsA = labels[mask]
        else:
            self.domainB = data[mask]
            self.labelsB = labels[mask]
        
        if self.verbose > 0:
            print(f"remove_section completed - removed {end_idx - start_idx} points from domain {domain}")
            print(f"Domain {domain} now has {len(data[mask])} points")
    
#& Next data set
#TODO
class helix_and_line(ToyData):
    """
    This class generates the helix and line data.
    Domain A: Helix 3D
    Domain B: Line 2D

    Parameters:
    - n_samples: int, number of samples to generate
    - n_classes: int, number of classes to generate
    - verbose: int, verbosity level
    - seed: int, random seed
    - noise_percentage: float, percentage of points to add noise to
    - noise_scale: float, scale of the noise
    """
    
    def __init__(self, n_samples, n_classes = 3, verbose=1, seed=SEEDS[0], noise_percentage=0.2, noise_scale=1):
        if verbose > 0:
            print(f"""helix_and_line.__init__ called with n_samples={n_samples}, n_classes={n_classes}, verbose={verbose},
             seed={seed}, noise_percentage={noise_percentage}, noise_scale={noise_scale}""")
            
        self.n_classes = n_classes
        self.noise_percentage = noise_percentage
        self.noise_scale = noise_scale
        self.noisy_indicies_A = None
        self.noisy_indicies_B = None
        
        super().__init__(n_samples, verbose, seed)

        if verbose > 0:
            print(f"helix_and_line.__init__ completed")

    def generate_domainA(self):
        """
        This function generates the domain A data (3D helix: x, sin(x), cos(x)).
        """
        if self.verbose > 0:
            print(f"generate_domainA (helix) called with n_samples={self.n_samples}, n_classes={self.n_classes}")

        # Generate more points than needed to ensure even distribution
        extra_samples = self.n_samples * 2
        
        # Generate points for a 3D helix
        t = np.linspace(0, 4 * np.pi, extra_samples)
        x = t
        y = np.sin(t)
        z = np.cos(t)
        data = np.column_stack([x, y, z])

        # Add noise
        n_noisy = int(extra_samples * self.noise_percentage)
        self.noisy_indicies_A = np.random.choice(extra_samples, n_noisy, replace=False)
        noise = np.random.normal(0, self.noise_scale, (n_noisy, 3))
        data[self.noisy_indicies_A] += noise

        # Create even-sized classes based on t
        points_per_class = self.n_samples // self.n_classes
        normalized_t = (t - t.min()) / (t.max() - t.min())
        initial_labels = np.floor(normalized_t * self.n_classes).astype(int)
        initial_labels = np.minimum(initial_labels, self.n_classes - 1)

        # Select points evenly from each class
        final_data = []
        final_labels = []
        for class_idx in range(self.n_classes):
            class_points = data[initial_labels == class_idx]
            if len(class_points) >= points_per_class:
                selected_indices = np.random.choice(len(class_points), points_per_class, replace=False)
                final_data.append(class_points[selected_indices])
                final_labels.extend([class_idx] * points_per_class)
            else:
                n_duplicates = points_per_class - len(class_points)
                duplicated_indices = np.random.choice(len(class_points), n_duplicates, replace=True)
                final_data.append(class_points)
                final_data.append(class_points[duplicated_indices])
                final_labels.extend([class_idx] * points_per_class)
        final_data = np.vstack(final_data)
        final_labels = np.array(final_labels)
        if self.verbose > 0:
            print(f"generate_domainA completed - generated {len(final_data)} points")
            for i in range(self.n_classes):
                print(f"  Class {i}: {np.sum(final_labels == i)} points")
        return final_data, final_labels

    def generate_domainB(self):
        """
        This function generates the domain B data (a 3D line: x, 0, 0).
        """
        if self.verbose > 0:
            print(f"generate_domainB (line) called with n_samples={self.n_samples}, n_classes={self.n_classes}")

        # Generate more points than needed for even distribution
        extra_samples = self.n_samples * 2
        x = np.linspace(0, 4 * np.pi, extra_samples)
        y = np.zeros(extra_samples)
        z = np.zeros(extra_samples)
        data = np.column_stack([x, y, z])
        # Add noise to all coordinates
        n_noisy = int(extra_samples * self.noise_percentage)
        self.noisy_indicies_B = np.random.choice(extra_samples, n_noisy, replace=False)
        noise = np.random.normal(0, self.noise_scale, (n_noisy, 3))
        data[self.noisy_indicies_B] += noise
        # Create even-sized classes based on x-coordinate
        points_per_class = self.n_samples // self.n_classes
        normalized_x = (x - x.min()) / (x.max() - x.min())
        initial_labels = np.floor(normalized_x * self.n_classes).astype(int)
        initial_labels = np.minimum(initial_labels, self.n_classes - 1)
        # Select points evenly from each class
        final_data = []
        final_labels = []
        for class_idx in range(self.n_classes):
            class_points = data[initial_labels == class_idx]
            if len(class_points) >= points_per_class:
                selected_indices = np.random.choice(len(class_points), points_per_class, replace=False)
                final_data.append(class_points[selected_indices])
                final_labels.extend([class_idx] * points_per_class)
            else:
                n_duplicates = points_per_class - len(class_points)
                duplicated_indices = np.random.choice(len(class_points), n_duplicates, replace=True)
                final_data.append(class_points)
                final_data.append(class_points[duplicated_indices])
                final_labels.extend([class_idx] * points_per_class)
        final_data = np.vstack(final_data)
        final_labels = np.array(final_labels)
        if self.verbose > 0:
            print(f"generate_domainB completed - generated {len(final_data)} points")
            for i in range(self.n_classes):
                print(f"  Class {i}: {np.sum(final_labels == i)} points")
        return final_data, final_labels
    
#& Shared and Private Regions for Manifold Alignment
class AlienRegions(ToyData):
    """
    This class generates two domains, each with n_shared_clusters shared clusters and one private cluster.
    Shared clusters: Gaussian blobs at fixed positions in both domains (Domain B's are shifted).
    Private region A: Only in Domain A, blob at a unique location.
    Private region B: Only in Domain B, blob at a different unique location.
    Labels: 0..n_shared_clusters-1=shared, n_shared_clusters=privateA, n_shared_clusters+1=privateB.
    Each region has the same number of points.
    """
    def __init__(self, n_samples, n_shared_clusters=3, std=0.4, verbose=1, seed=SEEDS[0]):
        self.std = std
        self.n_shared_clusters = n_shared_clusters
        super().__init__(n_samples, verbose, seed)
        if verbose > 0:
            print(f"AlienRegions.__init__ completed with n_samples={n_samples}, n_shared_clusters={n_shared_clusters}, std={std}, seed={seed}")

    def generate_domainA(self):
        np.random.seed(self.seed)
        n_regions = self.n_shared_clusters + 1  # shared + private
        n_per_region = self.n_samples // n_regions
        data = []
        labels = []
        # Shared clusters: arrange on a circle
        for i in range(self.n_shared_clusters):
            angle = 2 * np.pi * i / self.n_shared_clusters
            center = np.array([2 * np.cos(angle), 2 * np.sin(angle)])
            points = np.random.normal(loc=center, scale=self.std, size=(n_per_region, 2))
            data.append(points)
            labels.extend([i] * n_per_region)
        # Private cluster for A
        private_center = np.array([5, 0])
        private_points = np.random.normal(loc=private_center, scale=self.std, size=(n_per_region, 2))
        data.append(private_points)
        labels.extend([self.n_shared_clusters] * n_per_region)
        data = np.vstack(data)
        labels = np.array(labels)
        return data, labels

    def generate_domainB(self):
        np.random.seed(self.seed)
        n_regions = self.n_shared_clusters + 1  # shared + private
        n_per_region = self.n_samples // n_regions
        data = []
        labels = []
        # Shared clusters: arrange on a circle, but shifted
        shift = np.array([-2, 2])
        for i in range(self.n_shared_clusters):
            angle = 2 * np.pi * i / self.n_shared_clusters
            center = np.array([2 * np.cos(angle), 2 * np.sin(angle)]) + shift
            points = np.random.normal(loc=center, scale=self.std, size=(n_per_region, 2))
            data.append(points)
            labels.extend([i] * n_per_region)
        # Private cluster for B
        private_center = np.array([-5, 0])
        private_points = np.random.normal(loc=private_center, scale=self.std, size=(n_per_region, 2))
        data.append(private_points)
        labels.extend([self.n_shared_clusters + 1] * n_per_region)
        data = np.vstack(data)
        labels = np.array(labels)
        return data, labels 

    def remove_alien_region(self):
        """
        Remove the "alien" (private) region from both domainA and domainB.

        In this context, each domain consists of several clusters:
            - `n_shared_clusters` clusters are shared between both domains.
            - Each domain has one additional private cluster, called the "alien" region.

        This method removes all points belonging to the alien region from both domains:
            - For domainA, the alien label is `n_shared_clusters`.
            - For domainB, the alien label is `n_shared_clusters + 1`.

        Returns:
            ((removed_points_A, removed_labels_A), (removed_points_B, removed_labels_B)):
                - removed_points_A: np.ndarray of points removed from domainA
                - removed_labels_A: np.ndarray of corresponding labels from domainA
                - removed_points_B: np.ndarray of points removed from domainB
                - removed_labels_B: np.ndarray of corresponding labels from domainB

        After calling this method, the alien regions are no longer present in self.domainA/self.labelsA and self.domainB/self.labelsB.
        """
        label = self.n_shared_clusters
        if self.verbose > 0:
            print(f"[remove_alien_region] Removing alien regions: n_samples={self.n_samples}, n_shared_clusters={self.n_shared_clusters}, alien_label_A={label}, alien_label_B={label + 1}")
        # Find indices to remove
        maskA = self.labelsA == label
        maskB = self.labelsB == (label + 1)
        removed_points_A = self.domainA[maskA]
        removed_labels_A = self.labelsA[maskA]
        removed_points_B = self.domainB[maskB]
        removed_labels_B = self.labelsB[maskB]
        # Remove from domains
        self.domainA = self.domainA[~maskA]
        self.labelsA = self.labelsA[~maskA]
        self.domainB = self.domainB[~maskB]
        self.labelsB = self.labelsB[~maskB]
        if self.verbose > 0:
            print(f"[remove_alien_region] Completed: removed {len(removed_labels_A)} points from domainA and {len(removed_labels_B)} points from domainB.")
            
        return (removed_points_A, removed_labels_A), (removed_points_B, removed_labels_B)

#& MNIST Domains for Manifold Alignment
class MNISTDomains(ToyData):
    """
    This class generates two domains from MNIST data with specified shared and private digits.
    
    Parameters:
    - shared_digits: list of digits (0-9) that appear in both domains
    - private_digits_A: list of digits (0-9) that appear only in domain A
    - private_digits_B: list of digits (0-9) that appear only in domain B
    - samples_per_digit: number of samples to use per digit
    - use_pca: whether to apply PCA for dimensionality reduction (default: True)
    - n_components: number of PCA components if use_pca=True (default: 50)
    """
    
    def __init__(self, shared_digits, private_digits_A, private_digits_B, 
                 samples_per_digit=100, use_pca=True, n_components=50, 
                 verbose=1, seed=SEEDS[0]):
        if verbose > 0:
            print(f"MNISTDomains.__init__ called with shared_digits={shared_digits}, "
                  f"private_digits_A={private_digits_A}, private_digits_B={private_digits_B}, "
                  f"samples_per_digit={samples_per_digit}, use_pca={use_pca}, "
                  f"n_components={n_components}, seed={seed}")
        
        self.shared_digits = shared_digits
        self.private_digits_A = private_digits_A
        self.private_digits_B = private_digits_B
        self.samples_per_digit = samples_per_digit
        self.use_pca = use_pca
        self.n_components = n_components
        
        # Calculate total samples
        n_shared = len(shared_digits) * samples_per_digit
        n_private_A = len(private_digits_A) * samples_per_digit
        n_private_B = len(private_digits_B) * samples_per_digit
        self.n_samples = max(n_shared + n_private_A, n_shared + n_private_B)
        
        super().__init__(self.n_samples, verbose, seed)
        
        if verbose > 0:
            print(f"MNISTDomains.__init__ completed")
    
    def _load_mnist_data(self):
        """Load and preprocess MNIST data."""
        from sklearn.datasets import fetch_openml
        from sklearn.preprocessing import StandardScaler
        
        if self.verbose > 0:
            print("Loading MNIST data...")
        
        # Load MNIST data
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        
        # Normalize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if self.use_pca:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.n_components, random_state=self.seed)
            X_reduced = pca.fit_transform(X_scaled)
            if self.verbose > 0:
                print(f"Applied PCA: {X.shape[1]} -> {X_reduced.shape[1]} dimensions")
            return X_reduced, y
        else:
            return X_scaled, y
    
    def _sample_digits(self, X, y, digits, n_samples_per_digit):
        """Sample specified number of points for each digit."""
        data = []
        labels = []
        
        for digit in digits:
            digit_indices = np.where(y == digit)[0]
            if len(digit_indices) < n_samples_per_digit:
                # If not enough samples, use all available
                selected_indices = digit_indices
                if self.verbose > 0:
                    print(f"Warning: Only {len(digit_indices)} samples available for digit {digit}, "
                          f"requested {n_samples_per_digit}")
            else:
                # Randomly sample
                np.random.seed(self.seed + digit)  # Different seed for each digit
                selected_indices = np.random.choice(digit_indices, n_samples_per_digit, replace=False)
            
            data.append(X[selected_indices])
            labels.extend([digit] * len(selected_indices))
        
        return np.vstack(data), np.array(labels)
    
    def generate_domainA(self):
        """Generate domain A data with shared and private digits."""
        if self.verbose > 0:
            print(f"generate_domainA called")
        
        X, y = self._load_mnist_data()
        
        # Get all digits for domain A
        all_digits_A = self.shared_digits + self.private_digits_A
        
        # Sample data for domain A
        data_A, labels_A = self._sample_digits(X, y, all_digits_A, self.samples_per_digit)
        
        if self.verbose > 0:
            print(f"generate_domainA completed - generated {len(data_A)} points")
            for digit in all_digits_A:
                count = np.sum(labels_A == digit)
                print(f"  Digit {digit}: {count} points")
        
        return data_A, labels_A
    
    def generate_domainB(self):
        """Generate domain B data with shared and private digits."""
        if self.verbose > 0:
            print(f"generate_domainB called")
        
        X, y = self._load_mnist_data()
        
        # Get all digits for domain B
        all_digits_B = self.shared_digits + self.private_digits_B
        
        # Sample data for domain B
        data_B, labels_B = self._sample_digits(X, y, all_digits_B, self.samples_per_digit)
        
        if self.verbose > 0:
            print(f"generate_domainB completed - generated {len(data_B)} points")
            for digit in all_digits_B:
                count = np.sum(labels_B == digit)
                print(f"  Digit {digit}: {count} points")
        
        return data_B, labels_B