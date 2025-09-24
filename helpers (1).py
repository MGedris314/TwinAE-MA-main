# Helper Functions
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

#& Display Random Images
def display_random_image(features, labels, dataset_info, n_images=1):
    """
    Display random images from a dataset.
    
    Args:
        features (np.array): Feature array containing image data
        labels (np.array): Label array
        dataset_info (dict): Dataset information dictionary
        n_images (int): Number of random images to display
    """
    fig, axes = plt.subplots(1, n_images, figsize=(4*n_images, 4))
    if n_images == 1:
        axes = [axes]
    
    for i in range(n_images):
        # Select random image
        idx = np.random.randint(0, len(features))
        image_data = features[idx]
        label = labels[idx]
        
        # Reshape image based on dataset info
        img_shape = dataset_info['img_shape']
        
        if len(img_shape) == 3:  # Color image (e.g., CIFAR-10)
            # Reshape to (height, width, channels)
            image = image_data.reshape(img_shape)
            # Normalize to [0, 1] if values are in [0, 255] range
            if image.max() > 1:
                image = image / 255.0
            axes[i].imshow(image)
        elif len(img_shape) == 2:  # Grayscale image (e.g., MNIST)
            image = image_data.reshape(img_shape)
            # Normalize to [0, 1] if values are in [0, 255] range
            if image.max() > 1:
                image = image / 255.0
            axes[i].imshow(image, cmap='gray')
        else:
            # If we can't determine shape, show as 1D plot
            axes[i].plot(image_data)
            axes[i].set_title(f"1D representation")
        
        axes[i].set_title(f"{dataset_info['name']} - Label: {label}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

#& Load Image data
def load_image_dataset(dataset_name, data_path="../Data/toy_data/classification/"):
    """
    Load an image classification dataset from CSV file.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'CIFAR_10', 'Fashion-MNIST', 'mnist_784')
        data_path (str): Path to the data directory
    
    Returns:
        tuple: (features, labels, dataset_info)
    """
    file_path = Path(data_path) / f"{dataset_name}.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {file_path}")
    
    print(f"Loading {dataset_name} dataset...")
    df = pd.read_csv(file_path)
    
    # For image datasets, typically the last column is the label
    features = df.iloc[:, 1:].values
    labels = df.iloc[:, 0].values
    
    # Determine image dimensions based on feature count
    n_features = features.shape[1]
    if n_features == 784:  # 28x28 images (MNIST, Fashion-MNIST)
        img_shape = (28, 28)
    elif n_features == 3072:  # 32x32x3 images (CIFAR-10)
        img_shape = (32, 32, 3)
    else:
        # Try to find a square image dimension
        img_size = int(np.sqrt(n_features))
        if img_size * img_size == n_features:
            img_shape = (img_size, img_size)
        else:
            img_shape = (n_features,)  # 1D if can't determine
    
    dataset_info = {
        'name': dataset_name,
        'n_samples': len(df),
        'n_features': n_features,
        'n_classes': len(np.unique(labels)),
        'img_shape': img_shape,
        'class_labels': np.unique(labels)
    }
    
    print(f"Loaded {dataset_info['n_samples']} samples with {dataset_info['n_features']} features")
    print(f"Number of classes: {dataset_info['n_classes']}")
    print(f"Image shape: {dataset_info['img_shape']}")
    
    return features, labels, dataset_info

#& Map Manifold Space Functions
# Smooth animation function to traverse along the alpha shape boundary
def traverse_shape_boundary_smooth(emb, emb_labels, num_points=100, autoencoder=None):
    """
    Smoothly traverse along the boundary of a shapely geometry and generate images.
    
    Parameters:
    shape: Shapely geometry (Polygon or MultiPolygon)
    num_points: Number of points to sample along the boundary
    autoencoder: Autoencoder to use for image generation
    
    Returns:
    boundary_points: Array of points along the boundary
    images: List of generated images
    """
    from shapely.geometry import MultiPolygon, Polygon
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython.display import display, clear_output
    import time

    import alphashape
    from descartes import PolygonPatch

    # Compute the alpha shape (concave hull)
    alpha = 15 # You may need to tune this parameter
    shape = alphashape.alphashape(emb, alpha)

    #plot the shape
    # Plot the embedding points and the alpha shape
    plt.figure(figsize=(8, 6))
    plt.scatter(emb[:, 0], emb[:, 1], c=emb_labels, cmap='tab10', s=20, alpha=0.7)

    # Add the alpha shape boundary
    if isinstance(shape, MultiPolygon):
        for poly in shape.geoms:
            if poly.is_valid and not poly.is_empty:
                patch = PolygonPatch(poly, alpha=0.3, color='orange')
                plt.gca().add_patch(patch)
    elif isinstance(shape, Polygon):
        if shape.is_valid and not shape.is_empty:
            patch = PolygonPatch(shape, alpha=0.3, color='orange')
            plt.gca().add_patch(patch)

    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.title(f'Alpha Shape Boundary (Alpha = {alpha})')
    plt.colorbar(label='Class')
    plt.grid(alpha=0.3)
    plt.show()

    input("Press any key to Continue")

    # Extract boundary coordinates
    if isinstance(shape, MultiPolygon):
        boundaries = []
        for poly in shape.geoms:
            if hasattr(poly, 'exterior') and poly.exterior:
                coords = list(poly.exterior.coords)
                if len(coords) > 1:
                    boundaries.extend(coords[:-1])
    elif isinstance(shape, Polygon):
        if hasattr(shape, 'exterior') and shape.exterior:
            coords = list(shape.exterior.coords)
            if len(coords) > 1:
                boundaries = coords[:-1]
            else:
                boundaries = []
        else:
            boundaries = []
    else:
        print("Unsupported shape type")
        return None, None
    
    if not boundaries:
        print("No valid boundary found")
        return None, None
    
    print(f"Found {len(boundaries)} boundary points")
    
    # Convert to numpy array
    boundaries = np.array(boundaries)
    
    # Create a smooth path along the boundary
    from scipy.interpolate import interp1d
    
    # Calculate cumulative distance along the boundary
    distances = np.cumsum(np.sqrt(np.sum(np.diff(boundaries, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)
    
    # Create interpolation functions for x and y coordinates
    if len(distances) > 1:
        fx = interp1d(distances, boundaries[:, 0], kind='linear', bounds_error=False, fill_value='extrapolate')
        fy = interp1d(distances, boundaries[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # Generate evenly spaced distances
        total_length = distances[-1]
        sample_distances = np.linspace(0, total_length, num_points)
        
        # Interpolate boundary points
        boundary_points = np.column_stack([fx(sample_distances), fy(sample_distances)])
    else:
        boundary_points = np.tile(boundaries[0], (num_points, 1))
    
    # Generate images if autoencoder is provided
    images = []
    if autoencoder is not None:
        print(f"Generating {num_points} images along the shape boundary...")
        
        # Create the figure once
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Pre-plot the static elements
        ax1.scatter(emb[:, 0], emb[:, 1], c=emb_labels, cmap='tab10', s=20, alpha=0.7)
        
        # Plot the alpha shape
        try:
            if isinstance(shape, MultiPolygon):
                for poly in shape.geoms:
                    if poly.is_valid and not poly.is_empty:
                        patch = PolygonPatch(poly, alpha=0.3, color='orange')
                        ax1.add_patch(patch)
            elif isinstance(shape, Polygon):
                if shape.is_valid and not shape.is_empty:
                    patch = PolygonPatch(shape, alpha=0.3, color='orange')
                    ax1.add_patch(patch)
        except Exception as e:
            print(f"Warning: Could not plot shape: {e}")
            ax1.plot(boundaries[:, 0], boundaries[:, 1], 'orange', linewidth=2, alpha=0.7)
        
        ax1.set_xlabel('Embedding Dimension 1')
        ax1.set_ylabel('Embedding Dimension 2')
        ax1.set_title('Boundary Point Location')
        
        # Set up the image subplot
        ax2.set_title('Generated Image')
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Animation loop
        for i, point in enumerate(boundary_points):
            # Generate image from the boundary point
            artificial_img = autoencoder.inverse_transform(point.reshape(1, -1)).reshape(28, 28)
            images.append(artificial_img)
            
            # Update the boundary point marker (remove old one, add new one)
            ax1.scatter(point[0], point[1], c='red', s=100, marker='*', edgecolors='black', linewidth=2)
            ax1.set_title(f'Boundary Point {i+1}/{num_points}\nCoordinates: ({point[0]:.3f}, {point[1]:.3f})')

            
            # Update the image
            ax2.imshow(artificial_img, cmap='gray')
            ax2.set_title(f'Generated Image {i+1}')
            
            # Update the display
            display(fig)
            clear_output(wait=True)
            
            # Small delay for smooth animation
            time.sleep(0.01)  # Reduced from 0.1 to 0.05 for smoother animation
        
        plt.close(fig)  # Close the figure to free memory
    
    return boundary_points, images