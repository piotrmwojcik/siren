import copy

import torch
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import modules, dataio, nerf_utils
from torch.utils.data import DataLoader
import configargparse
import matplotlib.pyplot as plt


exp_name = "test1"
base_dir = os.path.join(os.path.dirname(os.getcwd()), "nerf_experiments", exp_name)

checkpoint_path = f'/Users/kacpermarzol/PycharmProjects/siren2/siren/nerf_experiments/{exp_name}/checkpoints/epoch8000.pth'
weights = torch.load(checkpoint_path, map_location=torch.device('cpu'))

model = modules.SingleBVPNet(out_features=4, in_features=3, mode='nerf')
model.load_state_dict(weights)

sdf_dataset = dataio.SRNDatasetsLMDB("cars", dataset_root="/Users/kacpermarzol/Downloads/datasets", split='test')
dataloader = DataLoader(sdf_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=0)
dataloader_iterator = iter(dataloader)

# spiral = sdf_dataset[0]
# spiral_input = spiral[0]
# spiral_gt = spiral[1]
# spiral_input['focal'] = torch.tensor([spiral_input['focal']])
# spiral_gt['img'] = spiral_gt['img'].unsqueeze(0)

p = configargparse.ArgumentParser()
opt = p.parse_args()
opt.subsampled_views=0
opt.rendering_type = 'baseline'
opt.H, opt.W = 128,128
opt.num_samples_per_ray = 31
opt.near = 0.8
opt.far = 1.8
opt.lindisp = sdf_dataset.lindisp
opt.white_bkgd = True
opt.randomized = False
opt.rgb_activation = 'sigmoid'
opt.density_activation = 'elu'

spiral_input, spiral_gt = next(dataloader_iterator)
focal = spiral_input['focal']
spiral_gt = {'img': spiral_gt['img'][0][0].unsqueeze(0).unsqueeze(0)}

gif = []
for i in range(250):
    model_input = {'idx': torch.tensor([0]), 'focal': focal, 'c2w': spiral_input['c2w'][0][i].unsqueeze(0).unsqueeze(0)}
    model_input_eval, gt_eval = nerf_utils.get_samples_for_nerf(model_input, copy.deepcopy(spiral_gt), opt, view_sampling=False, pixel_sampling=False)
    model_output = model(model_input_eval)
    model_output = nerf_utils.nerf_volume_rendering(model_output, opt)
    img = nerf_utils.save_rendering_spiral(model_output, opt).squeeze(0)
    # img = img.permute(1, 2, 0).numpy()
    # plt.imshow(img)
    # plt.show()
    gif.append(img)

nerf_utils.create_gif(gif, os.path.join(base_dir, 'eval.gif'))


