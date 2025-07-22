#This is just a copy of the .ipynb file to make the import a little better.

import numpy as np
import pandas as pd

def generate_line_and_helix(n_points):
    """
    Generate two datasets:
    - Straight line: (x, y) where y = x
    - Helix: (x, y, z) where x matches the line, y = sin(x), z = cos(x)
    Returns:
        line_df: pd.DataFrame with columns ['x', 'y']
        helix_df: pd.DataFrame with columns ['x', 'y', 'z']
    """
    x = np.linspace(0, 4 * np.pi, n_points)
    helix_y = np.sin(x)
    helix_z = np.cos(x)
    line_df = pd.DataFrame({'x': x, 'y': 0})
    helix_df = pd.DataFrame({'x': x, 'y': helix_y, 'z': helix_z})
    return line_df, helix_df

def generate_line_and_helix_with_noise(n_points, noise_percentage=0.2, noise_scale=3.0):
    """
    Generate two datasets with noise:
    - Straight line: (x, y) where y = x, with noise added to a percentage of points
    - Helix: (x, y, z) where x matches the line, y = sin(x), z = cos(x), with noise added to a percentage of points
    Args:
        n_points: Number of points to generate
        noise_percentage: Fraction of points to add noise to (0.0 to 1.0)
        noise_scale: Standard deviation of the noise to add
    Returns:
        line_df: pd.DataFrame with columns ['x', 'y']
        helix_df: pd.DataFrame with columns ['x', 'y', 'z']
    """
    x = np.linspace(0, 4 * np.pi, n_points)
    line_y = x.copy()
    helix_y = np.sin(x)
    helix_z = np.cos(x)

    n_noisy = int(n_points * noise_percentage)
    noisy_indices = np.random.choice(n_points, n_noisy, replace=False)

    # Add noise to the line
    line_y_noisy = line_y.copy()
    line_y_noisy[noisy_indices] += np.random.normal(0, noise_scale, n_noisy)

    # Add noise to the helix
    helix_y_noisy = helix_y.copy()
    helix_z_noisy = helix_z.copy()
    helix_y_noisy[noisy_indices] += np.random.normal(0, noise_scale, n_noisy)
    helix_z_noisy[noisy_indices] += np.random.normal(0, noise_scale, n_noisy)

    line_df = pd.DataFrame({'x': x, 'y': 0})
    helix_df = pd.DataFrame({'x': x, 'y': helix_y_noisy, 'z': helix_z_noisy})
    return line_df, helix_df, noisy_indices

