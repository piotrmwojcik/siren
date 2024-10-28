'''Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement.
'''

# Enable import from parent package
import sys
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse


class NerfObject(Dataset):
    def __init__(self, idx, focal, c2w, img):
        self.idx = idx
        self.focal = focal
        self.c2w = c2w
        self.img = img

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        in_dict = {'idx': idx, 'focal': self.focal, 'c2w': self.c2w}
        gt_dict = {'img': self.img}
        return in_dict, gt_dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, default="snrcars",  # required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1400)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--point_cloud_path', type=str, default='/home/sitzmann/data/point_cloud.xyz',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()

sdf_dataset = dataio.SRNDatasetsLMDB("cars", dataset_root="C:\\Personal\\siren\\data")

for sample_idx, sample in enumerate(sdf_dataset):
    in_dict, gt_dict = sample
    print(in_dict["c2w"].shape)
    print(gt_dict["img"].shape)
    break

# # Define the model.
# if opt.model_type == 'nerf':
#     model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3)
# else:
#     model = modules.SingleBVPNet(type=opt.model_type, in_features=3)
# model.cuda()
#
# # Define the loss
# loss_fn = loss_functions.sdf
# summary_fn = utils.write_sdf_summary
#
# root_path = os.path.join(opt.logging_root, opt.experiment_name)
#
# training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
#                steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
#                model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
#                clip_grad=True)
