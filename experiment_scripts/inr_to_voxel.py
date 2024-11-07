import sys, os
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import torch
import numpy as np
from skimage import measure
import trimesh

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import modules
import dataio
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

save_path = '/data/pwojcik/siren/random_mod/B3.pth'
B = torch.load(save_path)

# coords_full_grid = torch.load('/Users/kacpermarzol/PycharmProjects/siren2/siren/in_dict_shapenet_voxel.pth')['coords']
#coords_hmm = torch.load('./coords_dataset.pth')['coords']
#input_hmm = {
#        'idx': torch.tensor([0]),
#        'coords': coords_hmm.unsqueeze(0)
#    }

def dec2bin(x, bits):
    """
    Convert decimal to binary.

    Args:
        x (Tensor): Input tensor.
        bits (int): Number of bits for conversion.

    Returns:
        Tensor: Binary representation of the input tensor.
    """
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte().flip(-1)

def generate_mlp_from_weights(weights):
    mlp = modules.ImplicitMLP(B = B)

    state_dict = mlp.state_dict()
    weight_names = list(state_dict.keys())
    for layer in weight_names:
        val = state_dict[layer]
        num_params = np.product(list(val.shape))
        w = weights[:num_params]
        w = w.view(*val.shape)
        state_dict[layer] = w
        weights = weights[num_params:]
    assert len(weights) == 0, f"len(weights) = {len(weights)}"
    mlp.load_state_dict(state_dict)
    return mlp

def generate_input(coords_full_grid, i):
    idx_tensor = torch.tensor([0])
    coords = coords_full_grid[i * 16384 : (i+1) * 16384].transpose(1,0).view(3,128,128)
    input = {
        'idx': idx_tensor,
        'coords':  coords.unsqueeze(0)
    }

    return input

if __name__ == '__main__':
    # path = '/Users/__name__kacpermarzol/PycharmProjects/siren2/siren/logs/shapenet_voxel_sample500in_basic_rep_szum/0/ours/checkpoints/model_final.pth'
    path = '/data/pwojcik/siren/logs/TEST_DATASET2/0/ours/checkpoints/model_final.pth'
    weights = torch.load(path)
    model = modules.ImplicitMLP3D(B = B)
    model.load_state_dict(weights)
    results = []

    grid_size = 64

    coords = dataio.get_mgrid(grid_size, 3) / 2
    # coords = torch.from_numpy(torch.load('/Users/kacpermarzol/PycharmProjects/siren2/siren/coords_dataset.pth'))
    # coords += 0.01
    # results.append(model({
    #     'idx': 0,
    #     'coords':  coords.permute(1,0).view(3,128,128).unsqueeze(0)
    # })['model_out'].detach().squeeze(0).round())
    #
    for i in range(grid_size ** 3 // 16384):
        input = generate_input(coords, i)
        output = model(input)['model_out'].detach().squeeze(0).round()
        results.append(output)
    #
    results = torch.cat(results)

    grid = np.zeros((grid_size, grid_size, grid_size))

    print('!!!')
    print(grid.shape)

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(
            np.array(results.numpy()), level=0.5, spacing=[64] * 3
        )
    except Exception as e:
        print('Marching cubes failed! ', e)


    coords = ((coords + 0.5) * (grid_size - 1)).numpy().round() ### !!! ten round bardzo wazny
    indices = np.clip(coords.astype(int), 0, grid_size - 1)
    # do wizualizacji GT
    # results = torch.load('/Users/kacpermarzol/PycharmProjects/siren2/siren/img.pth') ## GT
    # results = torch.load('/Users/kacpermarzol/PycharmProjects/siren2/siren/img2.pth')

    # results=torch.ones_like(output)
    # results[results ==  0] = -1
    # results[results ==  1 ] = 0
    # results[results ==  -1 ] = 1
    unique_values, counts = np.unique(results, return_counts=True)
    for value, count in zip(unique_values, counts):
        print(f"Value: {value}, Count: {count}")
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = results.flatten()
    #
    #
    # verts, faces, _, _ = measure.marching_cubes(grid, level=0.5)
    # mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # mesh.show()

    threshold = 0.5
    voxel_data = grid > threshold  # Boolean array for voxels above threshold

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Use plt.voxels to plot
    ax.voxels(voxel_data, facecolors='cyan', edgecolor=None)

    # Labels and plot display
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    plt.show()

    # Save the plot as an image file (e.g., PNG)
    save_path = "voxel_plot.png"  # Define the desired save path and file name
    plt.savefig(save_path, format='png', dpi=300)
    print(f"Voxel plot saved as {save_path}")




    # filled = (results == 1)
    # filled_points = coords[filled.flatten()]
    # unfilled = (results == 0)
    # unfilled_points = coords[unfilled.flatten()]
    #
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.scatter(filled_points[:, 0], filled_points[:, 1], filled_points[:, 2],
    #            c='blue', s=1, label='Filled (1)')
    # #
    # # Plot unfilled points (value 0) in red
    # # ax.scatter(unfilled_points[:, 0], unfilled_points[:, 1], unfilled_points[:, 2],
    # #            c='red', s=1, label='Unfilled (0)')
    #
    # # Set plot labels and title
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('3D Point Cloud Visualization with Color-Coded Points')
    #
    # # Add legend
    # ax.legend(loc='upper right')
    #
    # # Optionally adjust the view angle (e.g., flipped view)
    # ax.view_init(elev=270, azim=90)  # Modify as needed
    #
    # plt.show()

