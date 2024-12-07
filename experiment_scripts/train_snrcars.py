'''Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement.
'''

# Enable import from parent package
import sys
import os
import time
import torch
import copy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio, meta_modules, utils, training, loss_functions, modules, nerf_utils

from torch.utils.data import DataLoader
import configargparse

exp_name = "test1"
base_dir = os.path.join(os.path.dirname(os.getcwd()), "nerf_experiments", exp_name)
rendering_dir = os.path.join(base_dir, "renders")
checkpoints_dir = os.path.join(base_dir, "checkpoints")

os.makedirs(rendering_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)

# class NerfObject(Dataset):
#     def __init__(self, idx, focal, c2w, img):
#         self.idx = idx
#         self.focal = focal
#         self.c2w = c2w
#         self.img = img
#
#     def __len__(self):
#         return 1
#
#     def __getitem__(self, idx):
#         in_dict = {'idx': idx, 'focal': self.focal, 'c2w': self.c2w}
#         gt_dict = {'img': self.img}
#         return in_dict, gt_dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, default="snrcars",  # required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=500,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='nerf',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--point_cloud_path', type=str, default='/home/sitzmann/data/point_cloud.xyz',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

# nerf
# NeRF strategy
p.add_argument('--epoch_for_full_rendering', type=int, default=100)
p.add_argument('--subsampled_views', type=int, default=32,
               help='Number of sampling views per scene when training')
p.add_argument('--subsampled_pixels', type=int, default=512,
               help='Number of sampling pixels per each view when training')
p.add_argument('--num_samples_per_ray', type=int, default=31,
               help='Number of points per each ray')
p.add_argument('--near', type=float, default=0.8)
p.add_argument('--far', type=float, default=1.8)
p.add_argument('--use_noise', action='store_true')
p.add_argument('--prob_mask_sampling', type=float, default=0)
p.add_argument('--rgb_activation', type=str, default='sigmoid')
p.add_argument('--density_activation', type=str, default='elu')
p.add_argument('--zero_to_one', action='store_true')
p.add_argument('--functa_rendering', action='store_true')
p.add_argument('--chuncking_unit', default=1, type=int)
p.add_argument('--rendering_type', type=str, default='baseline')

opt = p.parse_args()

sdf_dataset = dataio.SRNDatasetsLMDB("cars", dataset_root="/Users/kacpermarzol/Downloads/datasets")
dataloader = DataLoader(sdf_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=0)


# for sample_idx, sample in enumerate(sdf_dataset):
#     in_dict, gt_dict = sample
#
#     print(in_dict["c2w"].shape)
#     print(gt_dict["img"].shape)
#     break

# Define the model.
model = modules.SingleBVPNet(out_features=4, in_features=3, mode='nerf')
model.to(device)
optim = torch.optim.Adam(lr=opt.lr, params=model.parameters())


opt.randomized = opt.use_noise
opt.lindisp = sdf_dataset.lindisp
opt.white_bkgd = True
opt.H, opt.W = 128,128
in_channels, out_channels = 3, 4

loss_fn = loss_functions.image_mse
summary_fn = utils.write_sdf_summary
root_path = os.path.join(opt.logging_root, opt.experiment_name)


total_steps = 0
with tqdm(total=len(dataloader) * opt.num_epochs) as pbar:
    for epoch in range(opt.num_epochs):
        print(f'Epoch {epoch}/{opt.num_epochs}')
        all_psnr = 0.0
        steps = 0
        for step, (model_input_batch, gt_batch) in enumerate(dataloader):
            # print(step, model_input_batch.keys() ,gt_batch.keys()) # input: dict(idx, focal, c2w), gt: dict(img)
            if epoch % opt.epoch_for_full_rendering == 0 and step == 0:
                model_input_eval, gt_eval = nerf_utils.get_samples_for_nerf(copy.deepcopy(model_input_batch),
                                                                 copy.deepcopy(gt_batch), opt, view_num=1,
                                                                 pixel_sampling=False)
            model_input, gt = nerf_utils.get_samples_for_nerf(model_input_batch, gt_batch, opt)

            # print(model_input.keys(), gt.keys()) #input: dict(idx, coords, rays_d, t_vals), gt: dict(img)
            # print(model_input['coords'].shape, model_input['rays_d'].shape, model_input['t_vals'].shape) #torch.Size([1, 524288, 3]) torch.Size([16384, 3]) torch.Size([16384, 32])
            # print(gt['img'].shape) #torch.Size([1, 16384, 3])
            model_input = {key: value.to(device) for key, value in model_input.items()}
            gt = {key: value.to(device) for key, value in gt.items()}
            batch_size = gt['img'].size(0)
            model_output = model(model_input)

            model_output = nerf_utils.nerf_volume_rendering(model_output, opt)
            losses = loss_fn(model_output, gt)

            # PSNR
            for pred_img, gt_img in zip(model_output['model_out'].cpu(), gt['img'].cpu()):
                psnr = nerf_utils.compute_psnr(pred_img, gt_img)
                # psnr = compute_psnr(pred_img * 0.5 + 0.5, gt_img * 0.5 + 0.5)  # rescale from [-1, 1] to [0, 1]
                all_psnr += psnr
                steps += 1
            print(f'LOSS: {losses["img_loss"]}, PSNR: {psnr}')

            optim.zero_grad()
            for loss in losses.values():
                loss.backward()
            optim.step()


            if epoch % opt.epoch_for_full_rendering == 0 and step == 0:
                model_input_eval = {key: value.to(device) for key, value in model_input_eval.items()}
                gt_eval = {key: value.to(device) for key, value in gt_eval.items()}

                model_output_full = model(model_input_eval)
                model_output_full = nerf_utils.nerf_volume_rendering(model_output_full, opt, 'all')
                nerf_utils.save_rendering_output(model_output_full, gt_eval, opt,
                                      os.path.join(rendering_dir, f'E{epoch}_S{step}.png'))

            pbar.update(1)
            break

        if ((epoch % opt.epochs_til_ckpt == 0) or (epoch == opt.num_epochs - 1)):
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'epoch{epoch}.pth'))



