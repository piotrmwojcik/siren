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
import matplotlib.cm as cm

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
p.add_argument('--num_epochs_ours', type=int, default=5001)

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

writer_ours = SummaryWriter(summaries_dir_ours)

steps_ours = [500*i for i in range(opt.num_epochs_ours // 500 + 1)]

results_ours = None
counter = 0

lrs = [0.001, 0.003, 0.005, 0.007, 0.009, 0.01]
results = {lr: None for lr in lrs}

for png_file in jpg_files[:30]:
    counter += 1
    full_path = os.path.abspath(png_file)
    file_name = os.path.basename(png_file)
    print(f"Processing file: {full_path}")

    img_dataset = dataio.ImageFile(full_path)
    coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=64, compute_diff='none', grid='our')
    image_resolution = (64, 64)
    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)


    for lr in lrs:
        model_ours = modules.ImplicitMLP(B=B)
        model_ours.load_state_dict(torch.load('logs/112842.jpg/ours/checkpoints/model_final.pth', map_location=device))
        model_ours.to(device)


        root_path_ours = os.path.join(opt.logging_root, opt.experiment_name, file_name, 'ours', str(lr))
        root_path_siren = os.path.join(opt.logging_root, opt.experiment_name, file_name, 'siren')

        loss_fn = partial(loss_functions.image_mse, None)
        summary_fn = partial(utils.write_image_summary_empty, image_resolution)

        psnr_ours = training.train(model=model_ours, train_dataloader=dataloader, epochs=opt.num_epochs_ours, lr=lr,
                       steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                       model_dir=root_path_ours, loss_fn=loss_fn, summary_fn=summary_fn, device = device, writer=writer_ours)


        if results[lr] is not None:
            results[lr] = np.vstack((results[lr], np.array(psnr_ours)))
        else:
            results[lr] = np.array(psnr_ours)

colors = cm.viridis(np.linspace(0, 1, len(lrs)))

for i, lr in enumerate(lrs):
    mean_psnr = np.mean(results[lr], 0)
    std_psnr = np.std(results[lr], 0)

    plt.plot(steps_ours, mean_psnr, label=f"LR: {lr}", color=colors[i], marker='o')

    for j in range(len(steps_ours)):
        plt.text(steps_ours[j], mean_psnr[j], f"±{std_psnr[j]:.2f}", color=colors[i], fontsize=9)



plt.title(f"PSNR with Standard Deviations with Different Learning Rates for {counter} images")
plt.xlabel("Steps")
plt.ylabel("PSNR")
plt.legend()

plt.grid(True)
plt.savefig('psnr_lr.png', dpi=300)

