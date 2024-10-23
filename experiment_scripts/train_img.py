# Enable import from parent package
import sys
import os
import numpy as np
import glob
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import dataio, meta_modules, utils, training, loss_functions, modules
# https://github.com/DavideBuffelli/SAME/issues/1

from torch.utils.data import DataLoader
import configargparse
from functools import partial
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr_siren', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--lr_ours', type=float, default=5e-4)
p.add_argument('--num_epochs_siren', type=int, default=10001,
               help='Number of epochs to train for.')
p.add_argument('--num_epochs_ours', type=int, default=15001)

p.add_argument('--image_path', type=str, required=True,
               help='Path to the gt image.')

p.add_argument('--epochs_til_ckpt', type=int, default=500,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=500,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

opt = p.parse_args()

jpg_files = glob.glob(os.path.join(opt.image_path, "*.jpg"))
num_input_channels=2
mapping_dim=128
scale = 10
# B = torch.randn((num_input_channels, mapping_dim)) * scale
save_path = 'data/minidataset/B.pth'
# torch.save(B, save_path)
B = torch.load(save_path)

summaries_dir = os.path.join(opt.logging_root, opt.experiment_name, 'summary')
summaries_dir_siren = os.path.join(opt.logging_root, opt.experiment_name, 'summary', 'siren')
summaries_dir_ours = os.path.join(opt.logging_root, opt.experiment_name, 'summary', 'ours')

writer_siren = SummaryWriter(summaries_dir_siren)
writer_ours = SummaryWriter(summaries_dir_ours)

steps_siren = np.array([500*i for i in range(opt.num_epochs_siren // 500 + 1)])
steps_ours = np.array([500*i for i in range(opt.num_epochs_ours // 500 + 1)])

sum_psnr_siren = [0 for i in range(opt.num_epochs_siren // 500 + 1)]
sum_psnr_ours = [0 for i in range(opt.num_epochs_ours // 500 + 1)]

results_siren = None
results_ours = None

counter = 0
for png_file in jpg_files[:50]:
    counter += 1
    full_path = os.path.abspath(png_file)
    file_name = os.path.basename(png_file)
    print(f"Processing file: {full_path}")


    img_dataset = dataio.ImageFile(full_path)
    coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=64, compute_diff='none', grid='our')
    image_resolution = (64, 64)
    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

    model_ours = modules.ImplicitMLP(B=B)
    # model_ours.load_state_dict(torch.load('logs/002328.jpg/ours/checkpoints/model_final.pth', map_location=device))
    model_ours.to(device)

    img_dataset_siren = dataio.ImageFile(full_path)
    coord_dataset_siren = dataio.Implicit2DWrapper(img_dataset_siren, sidelength=64, compute_diff='none', grid='siren')
    image_resolution = (64, 64)

    dataloader_siren = DataLoader(coord_dataset_siren, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

    model_siren = modules.SingleBVPNet(sidelength=image_resolution, out_features = 3)
    model_siren.to(device)

    root_path_ours = os.path.join(opt.logging_root, opt.experiment_name, file_name, 'ours')
    root_path_siren = os.path.join(opt.logging_root, opt.experiment_name, file_name, 'siren')

    loss_fn = partial(loss_functions.image_mse, None)
    summary_fn = partial(utils.write_image_summary, image_resolution)

    psnr_ours = training.train(model=model_ours, train_dataloader=dataloader, epochs=opt.num_epochs_ours, lr_init=1e-3, lr_finish=1e-4,
                   steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                   model_dir=root_path_ours, loss_fn=loss_fn, summary_fn=summary_fn, device = device, writer=writer_ours)

    psnr_siren = training.train(model=model_siren, train_dataloader=dataloader_siren, epochs=opt.num_epochs_siren, lr_init=opt.lr_siren,
                   steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                   model_dir=root_path_siren, loss_fn=loss_fn, summary_fn=summary_fn, device=device,
                   writer=writer_siren)


    if results_siren is not None:
        results_siren = np.vstack((results_siren, np.array(psnr_siren)))
    else:
        results_siren = np.array(psnr_siren)

    if results_ours is not None:
        results_ours = np.vstack((results_ours, np.array(psnr_ours)))
    else:
        results_ours = np.array(psnr_ours)

mean_psnr_siren = np.mean(results_siren, 0)
mean_psnr_ours = np.mean(results_ours,0 )
std_psnr_siren = np.std(results_siren,0)
std_psnr_ours = np.std(results_ours, 0)

plt.plot(steps_siren, mean_psnr_siren, label='siren', color='orange', marker='o')
plt.plot(steps_ours, mean_psnr_ours, label='ours', color='blue', marker='o')

for i in range(len(steps_siren)):
    plt.text(steps_siren[i], mean_psnr_siren[i], f"±{std_psnr_siren[i]:.2f}", color='orange', fontsize=9)

for i in range(len(steps_ours)):
    plt.text(steps_ours[i], mean_psnr_ours[i], f"±{std_psnr_ours[i]:.2f}", color='blue', fontsize=9)

plt.xlabel('Steps')
plt.ylabel('PSNR')
plt.title(f'PSNR with Standard Deviations for {counter} images')
plt.legend()

plt.grid(True)

plt.savefig(f'psnr_{opt.experiment_name}', dpi=300)

