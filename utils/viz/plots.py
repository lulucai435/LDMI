import matplotlib
matplotlib.use('Agg')  # This is hacky (useful for running on VMs)
import matplotlib.pyplot as plt

# import matplotlib as mpl
import matplotlib.pyplot as plt
# plt.rcParams.update(plt.rcParamsDefault)
# plt.rcParams['text.usetex'] = False

import matplotlib.pyplot as plt
import numpy as np

def plot_voxels_batch_new(voxels, ncols=6, threshold=0.5, save_fig=''):
    """Plots batches of voxel grids in a grid layout.
    
    Args:
        voxels (torch.Tensor): Shape (batch_size, 1, voxel_res, voxel_res, voxel_res).
        ncols (int): Number of columns in the grid of images.
        threshold (float): Value above which to consider a voxel occupied.
        save_fig (str): If provided, saves the figure to this path.
    """
    batch_size = voxels.shape[0]
    voxel_res = voxels.shape[2]
    
    # Calculate the number of rows needed
    nrows = int(np.ceil(batch_size / ncols))
    
    # Create the figure
    fig = plt.figure(figsize=(ncols * 2, nrows * 2))  # Adjust figure size dynamically
    
    # Permutation to get a better angle of the voxel data
    voxels = voxels.permute(0, 1, 2, 4, 3)
    
    for i in range(batch_size):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        
        # Threshold the voxels to get occupied points
        voxels[i, 0][voxels[i, 0] > threshold] = 1
        voxels[i, 0][voxels[i, 0] <= threshold] = 0
        
        # Get coordinates of occupied voxels
        coords = voxels[i, 0].nonzero(as_tuple=False)
        values = voxels[i, 0][coords[:, 0], coords[:, 1], coords[:, 2]]
        
        # Plot the voxels
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            s=0.1,  # Size of points
            c=values,  # Color by value
            cmap='gray'  # Grayscale colormap
        )
        
        # Remove axis lines and background grid
        ax.set_axis_off()
        
        # Set limits of the plot
        ax.set_xlim(0, voxel_res)
        ax.set_ylim(0, voxel_res)
        ax.set_zlim(0, voxel_res)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Save or display the figure
    if save_fig:
        plt.savefig(save_fig, format='png', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.show()

def plot_voxels_batch(voxels, ncols=4, save_fig=''):
    """Plots batches of voxels.
    
    Args:
        voxels (torch.Tensor): Shape (batch_size, 1, depth, height, width).
        ncols (int): Number of columns in grid of images.
    """
    batch_size, _, voxel_size, _, _ = voxels.shape
    nrows = int((batch_size - 0.5) / ncols) + 1
    fig = plt.figure()
    
    # Permutation to get better angle of chair data
    voxels = voxels.permute(0, 1, 2, 4, 3)
    
    for i in range(batch_size):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')

        # Non zero voxels define coordinates of visible points
        coords = voxels[i, 0].nonzero(as_tuple=False)
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1)            
    
        # Set limits to size of voxel grid
        ax.set_xlim(0, voxel_size - 1)
        ax.set_ylim(0, voxel_size - 1)
        ax.set_zlim(0, voxel_size - 1)

    plt.tight_layout()
    
    # Optionally save figure
    if len(save_fig):
        plt.savefig(save_fig, format='png', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.show()


def plot_point_cloud_batch(point_clouds, ncols=4, threshold=0.5, save_fig=''):
    """Plots batches of point clouds
    
    Args:
        point_clouds (torch.Tensor): Shape (batch_size, num_points, 4).
        ncols (int): Number of columns in grid of images.
        threshold (float): Value above which to consider point cloud occupied.
    """
    batch_size = point_clouds.shape[0]
    nrows = int((batch_size - 0.5) / ncols) + 1
    fig = plt.figure()
    
    for i in range(batch_size):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')

        # Extract coordinates with feature values above threshold (corresponding
        # to occupied points)
        coordinates = point_clouds[i, :, :3]
        features = point_clouds[i, :, -1]
        coordinates = coordinates[features > threshold]
        ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], s=1)            
    
        # Set limits of plot
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

    plt.tight_layout()
    
    # Optionally save figure
    if len(save_fig):
        plt.savefig(save_fig, format='png', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.show()