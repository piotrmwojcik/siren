# Enable import from parent package
import sys
import os
import numpy as np
import glob
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataio, meta_modules, utils, training, loss_functions, modules
# https://github.com/DavideBuffelli/SAME/issues/1

from torch.utils.data import DataLoader
import configargparse
from functools import partial
from torch.utils.tensorboard import SummaryWriter
import h5py

class VoxelObject(Dataset):
    def __init__(self, idx, coords, img):
        self.idx = idx
        self.coords = coords
        self.img = img

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        in_dict = {'idx': idx, 'coords': self.coords}
        gt_dict = {'img': self.img}
        return in_dict, gt_dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, default='3d_voxel',
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr_siren', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--lr_ours', type=float, default=5e-4)
p.add_argument('--num_epochs_siren', type=int, default=10001,
               help='Number of epochs to train for.')
p.add_argument('--num_epochs_ours', type=int, default=10001)

# p.add_argument('--image_path', type=str, required=True,
#                help='Path to the gt image.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
# p.add_argument('--shapenet_path', type=str, default=None, required=True, help='Checkpoint to trained model.')

opt = p.parse_args()

num_input_channels = 3
mapping_dim = 128
scale = 10

# B = torch.randn((num_input_channels, mapping_dim)) * scale

save_path = 'data/minidataset/B2.pth'
# torch.save(B, save_path)
B = torch.load(save_path)
# shapenet = dataio.ShapeNetVoxel(dataset_root=opt.shapenet_path)

summaries_dir = os.path.join(opt.logging_root, opt.experiment_name, 'summary')
summaries_dir_siren = os.path.join(opt.logging_root, opt.experiment_name, 'summary', 'siren')
summaries_dir_ours = os.path.join(opt.logging_root, opt.experiment_name, 'summary', 'ours')

writer_siren = SummaryWriter(summaries_dir_siren)
writer_ours = SummaryWriter(summaries_dir_ours)

steps_siren = [1000 * i for i in range(opt.num_epochs_siren // 1000 + 1)]
steps_ours = [1000 * i for i in range(opt.num_epochs_ours // 1000 + 1)]

sum_psnr_siren = [0 for i in range(opt.num_epochs_siren // 1000 + 1)]
sum_psnr_ours = [0 for i in range(opt.num_epochs_ours // 1000 + 1)]

results_siren = None
results_ours = None

counter = 0

# sample = shapenet[0]
# in_dict, gt_dict = sample
# coords = in_dict['coords'] #zawsze takie same
# coords = ((coords + 1) / 2 * (64 - 1)).round()
dataset = h5py.File('/Users/kacpermarzol/PycharmProjects/hyperdiffusionproject/HyperDiffusion/siren/all_vox_tmp.hdf5', "r")


for sample_idx in range(3):
    counter += 1

    coords = dataset['points'][sample_idx]
    torch.save(coords, "coords_dataset.pth")
    img = dataset['occs'][sample_idx]
    x = VoxelObject(sample_idx, coords, img)

    print(f"Processing object: {sample_idx}")
    image_resolution = (64, 64, 64)
    dataloader_ours = DataLoader(x, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

    model_ours = modules.ImplicitMLP3D(B=B)

    model_ours.to(device)
    dataloader_siren = DataLoader(x, shuffle=True, batch_size=opt.batch_size,
                                  pin_memory=True, num_workers=0)
    image_resolution = (64, 64, 64)

    # model_siren = modules.SingleBVPNet(sidelength=image_resolution, out_features=1, in_features=3)
    # model_siren.to(device)

    root_path_ours = os.path.join(opt.logging_root, opt.experiment_name, str(sample_idx), 'ours')
    root_path_siren = os.path.join(opt.logging_root, opt.experiment_name, str(sample_idx), 'siren')

    # Define the loss
    loss_fn = partial(loss_functions.image_mse, None)
    summary_fn = partial(utils.write_image_summary, image_resolution)

    # to działa:
    # psnr_siren = training.train(model=model_siren, train_dataloader=dataloader_siren, epochs=opt.num_epochs_siren,
    #                             lr=opt.lr_siren,
    #                             steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
    #                             model_dir=root_path_siren, loss_fn=loss_fn, summary_fn=summary_fn, device=device,
    #                             writer=writer_siren)
    psnr_ours = training.train(model=model_ours, train_dataloader=dataloader_ours, epochs=opt.num_epochs_ours,
                               lr=opt.lr_ours,
                               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                               model_dir=root_path_ours, loss_fn=loss_fn, summary_fn=summary_fn, device=device,
                               writer=writer_ours, save_img=False)
    # if results_siren is not None:
    #     results_siren = np.vstack((results_siren, np.array(psnr_siren)))
    # else:
    #     results_siren = np.array(psnr_siren)

    if results_ours is not None:
        results_ours = np.vstack((results_ours, np.array(psnr_ours)))
    else:
        results_ours = np.array(psnr_ours)


    if sample_idx == 1:
        break


# mean_psnr_siren = np.mean(results_siren, 0)
# mean_psnr_ours = np.mean(results_ours, 0)
# std_psnr_siren = np.std(results_siren, 0)
# std_psnr_ours = np.std(results_ours, 0)

# for psnr, step in zip(mean_psnr_siren, steps_siren):
#     print(step, psnr)
#     writer_siren.add_scalar('psnr', psnr, step)
#
# for psnr, step in zip(mean_psnr_ours, steps_ours):
#     print(step, psnr)
#     writer_ours.add_scalar('psnr', psnr, step)


# import matplotlib.pyplot as plt

# plt.plot(steps_siren, mean_psnr_siren, label='siren', color='orange', marker='o')

# plt.plot(steps_ours, mean_psnr_ours, label='ours', color='blue', marker='o')

# for i in range(len(steps_siren)):
#     plt.text(steps_siren[i], mean_psnr_siren[i], f"±{std_psnr_siren[i]:.2f}", color='purple', fontsize=9)

# Annotate standard deviations for Ours
# for i in range(len(steps_ours)):
#     plt.text(steps_ours[i], mean_psnr_ours[i], f"±{std_psnr_ours[i]:.2f}", color='green', fontsize=9)
#
# plt.xlabel('Steps')
# plt.ylabel('PSNR')
# plt.title(f'PSNR with Standard Deviations for {counter} images')
# plt.legend()
#
# plt.grid(True)
#
# plt.savefig(f'exp_{opt.experiment_name}', dpi=300)
