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

coords_full_grid = torch.load('/Users/kacpermarzol/PycharmProjects/siren2/siren/in_dict_shapenet_voxel.pth')['coords']
coords_hmm = torch.load('/Users/kacpermarzol/PycharmProjects/siren2/siren/coords.pth')['coords']
input_hmm = {
        'idx': torch.tensor([0]),
        'coords': coords_hmm.unsqueeze(0)
    }

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

def generate_input(i):
    idx_tensor = torch.tensor([0])
    coords = coords_full_grid[i * 16384 : (i+1) * 16384 ].transpose(1,0).view(3,128,128)
    input = {
        'idx': idx_tensor,
        'coords':  ((coords + 1) / 2 * (64 - 1)).round().unsqueeze(0)
    }

    return input

if __name__ == '__main__':
    path = '/Users/kacpermarzol/PycharmProjects/siren2/siren/logs/cguk15/0/ours/checkpoints/model_final.pth'
    weights = torch.load(path)
    model = modules.ImplicitMLP3D(B = B)
    model.load_state_dict(weights)


    results = []

    for i in range(16):
        input = generate_input(i)
        output = model(input)['model_out'].detach().squeeze(0).round()
        results.append(output)
    results = torch.cat(results)

    grid_size = 64
    grid = np.zeros((grid_size, grid_size, grid_size))

    coords = coords_full_grid
    coords = ((coords + 1) / 2 * (grid_size - 1)).numpy().round() ### !!! ten round bardzo wazny
    indices = np.clip(coords.astype(int), 0, grid_size - 1)

    # do wizualizacji GT
    # results = torch.load('/Users/kacpermarzol/PycharmProjects/siren2/siren/img.pth') ## GT

    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = results.flatten()

    unique_values, counts = np.unique(results, return_counts=True)
    for value, count in zip(unique_values, counts):
        print(f"Value: {value}, Count: {count}")

    verts, faces, _, _ = measure.marching_cubes(grid, level=0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.show()

