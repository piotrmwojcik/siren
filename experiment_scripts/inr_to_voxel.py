import sys, os
import torch
import numpy as np
from skimage import measure
import trimesh

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import modules
import dataio
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


save_path = '/Users/kacpermarzol/PycharmProjects/siren2/siren/data/minidataset/B2.pth'
B = torch.load(save_path)

# num_input_channels = 3
# mapping_dim = 128
# scale = 10
# B = torch.randn((num_input_channels, mapping_dim)) * scale

coords = torch.load('/Users/kacpermarzol/PycharmProjects/siren2/siren/coords3.pth')
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

def generate_input():
    idx_tensor = torch.tensor([0])

    input = {
        'idx': idx_tensor,
        'coords': coords.permute(1, 0).view(3, 128, 128).unsqueeze(0)
    }

    return input

if __name__ == '__main__':
    # path = '/Users/kacpermarzol/PycharmProjects/siren2/siren/logs/TEST_VOXEL_NOGRID/0/ours/checkpoints/model_final.pth'
    # weights = torch.load(path)
    #
    # model = modules.ImplicitMLP3D(B = B)
    # model.load_state_dict(weights)
    #
    # input = generate_input()
    # output = model(input)
    # output = output['model_out'].detach().squeeze(0)
    # output = output.round().flatten()


    #with interpolaton


    output = torch.load('/Users/kacpermarzol/PycharmProjects/siren2/siren/img2.pth')

    grid_size = 64  # Adjust for resolution
    x = np.linspace(0, 63, grid_size)
    y = np.linspace(0, 63, grid_size)
    z = np.linspace(0, 63, grid_size)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing="ij")

    coords = ((coords + 1) / 2 * (grid_size - 1)).round().numpy()
    indices = np.clip(coords.astype(int), 0, grid_size - 1)
    grid_sdf = griddata(indices, output[:,0], (grid_x, grid_y, grid_z), method='linear', fill_value=-1)

    verts, faces, _, _ = measure.marching_cubes(grid_sdf, level=0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.show()


    # without interpolation

    # output = torch.load('/Users/kacpermarzol/PycharmProjects/siren2/siren/img2.pth')
    #
    # grid_size = 64
    # grid = np.zeros((grid_size, grid_size, grid_size))
    # coords = ((coords + 1) / 2 * (grid_size - 1)).round().numpy()
    # indices = np.clip(coords.astype(int), 0, grid_size - 1)
    #
    # # output[output == -1] = 0
    #
    # unique_values, counts = np.unique(output, return_counts=True)
    # for value, count in zip(unique_values, counts):
    #     print(f"Value: {value}, Count: {count}")
    #
    # grid[indices[:, 0], indices[:, 1], indices[:, 2]] = output.flatten()
    #
    # verts, faces, _, _ = measure.marching_cubes(grid, level=0)
    # mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # mesh.show()



