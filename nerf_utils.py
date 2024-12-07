import torch
import numpy as np
import math
import torch.nn.functional as F
import torchvision


def save_rendering_spiral_gif(model_output, opt, image_path, max_num=-1):
    print(model_output)
    pred_rgb = model_output['model_out']['rgb'].reshape(-1, opt.H, opt.W, 3).permute(0,3,1,2).detach().cpu()
    pred_depth = model_output['model_out']['depth'].reshape(-1, opt.H, opt.W, 1).permute(0,3,1,2).detach().cpu()
    pred_acc = model_output['model_out']['acc'].reshape(-1, opt.H, opt.W, 1).permute(0,3,1,2).detach().cpu()
    pred_depth = ((pred_depth-pred_depth.min())/(pred_depth.max()-pred_depth.min())*2-1).repeat(1,3,1,1)
    pred_acc = pred_acc.repeat(1,3,1,1)
    combined_image = torch.cat((pred_rgb, pred_depth, pred_acc), -1)

    if max_num > 0:
        save_num = min(combined_image.size(0), max_num)
        combined_image = combined_image[:save_num]

    combined_image = torchvision.utils.make_grid(combined_image, nrow=1)
    torchvision.utils.save_image(combined_image, image_path)

def save_rendering_output(model_output, gt, opt, image_path, max_num=-1):
    save_gt = gt['img'].reshape(-1, opt.H, opt.W, 3).permute(0,3,1,2).detach().cpu()
    pred_rgb = model_output['model_out']['rgb'].reshape(-1, opt.H, opt.W, 3).permute(0,3,1,2).detach().cpu()
    pred_depth = model_output['model_out']['depth'].reshape(-1, opt.H, opt.W, 1).permute(0,3,1,2).detach().cpu()
    pred_acc = model_output['model_out']['acc'].reshape(-1, opt.H, opt.W, 1).permute(0,3,1,2).detach().cpu()
    pred_depth = ((pred_depth-pred_depth.min())/(pred_depth.max()-pred_depth.min())*2-1).repeat(1,3,1,1)
    pred_acc = pred_acc.repeat(1,3,1,1)
    combined_image = torch.cat((save_gt, pred_rgb, pred_depth, pred_acc), -1)
    if max_num > 0:
        save_num = min(combined_image.size(0), max_num)
        combined_image = combined_image[:save_num]

    combined_image = torchvision.utils.make_grid(combined_image, nrow=1)
    torchvision.utils.save_image(combined_image, image_path)

def compute_psnr(signal, gt):
    mse = max(float(torch.mean((signal-gt)**2)), 1e-8)
    psnr = float(-10 * math.log10(mse))
    return psnr

def cast_rays(t_vals, origins, directions):
    return origins[..., None, :] + t_vals[..., None] * directions[..., None, :]

def sample_along_rays(
        cam_rays,
        configs,
):
    # get configs
    num_samples = configs.num_samples_per_ray
    near, far = configs.near, configs.far
    lindisp = configs.lindisp
    randomized = configs.randomized  # noise

    rays_o, rays_d = cam_rays[0], cam_rays[1]
    bsz = rays_o.shape[0]

    t_vals = torch.linspace(0.0, 1.0, num_samples + 1, device=rays_o.device)
    if lindisp:
        t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        t_vals = near * (1.0 - t_vals) + far * t_vals

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand((bsz, num_samples + 1), device=rays_o.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        t_vals = torch.broadcast_to(t_vals, (bsz, num_samples + 1))

    coords = cast_rays(t_vals, rays_o, rays_d)
    return t_vals, coords

def get_rays(H, W, focal, c2w, padding=None, compute_radii=False):
    # pytorch's meshgrid has indexing='ij'
    if padding is not None:
        i, j = torch.meshgrid(torch.linspace(-padding, W - 1 + padding, W + 2 * padding),
                              torch.linspace(-padding, H - 1 + padding, H + 2 * padding))
    else:
        i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    i = i.t().to(c2w.device)
    j = j.t().to(c2w.device)
    extra_shift = 0.5
    dirs = torch.stack([(i - W * .5 + extra_shift) / focal, -(j - H * .5 + extra_shift) / focal, -torch.ones_like(i)],
                       -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    if compute_radii:
        dx = torch.sqrt(torch.sum((rays_d[:-1, :, :] - rays_d[1:, :, :]) ** 2, -1))
        dx = torch.cat([dx, dx[-2:-1, :]], 0)

        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = dx[..., None] * 2 / math.sqrt(12)
        return torch.stack((rays_o, rays_d, radii.repeat(1, 1, 3)), 0)
    else:
        return torch.stack((rays_o, rays_d), 0)

def get_rays_batch(H, W, focal, c2w, compute_radii=False):
    # TODO: faster
    bsz = c2w.shape[0]
    all_rays = []
    for i in range(bsz):
        # per image
        cam_rays = get_rays(H, W, focal[i], c2w[i], compute_radii=compute_radii)
        all_rays.append(cam_rays)
    results = torch.stack(all_rays, 1)
    #results = torch.stack(list(map(lambda f, c: get_rays(H,W,f,c), focal, c2w)), 1)
    return results


def get_samples_for_nerf(model_input, gt, opt, view_sampling=True, pixel_sampling=True, view_num=None):
    all_scene_rays = []
    all_scene_rgb = []
    all_scene_idx = []
    ALL_VIEW = gt['img'].shape[1]
    bsz = gt['img'].shape[0]

    print(model_input['focal'])
    for i_batch in range(bsz):
        focal = model_input['focal'][i_batch]  # (ALL_VIEW)
        c2w = model_input['c2w'][i_batch]  # (ALL_VIEW,4,4)
        # idx_ = model_input['idx'][i_batch]  # 1
        rgb = gt['img'][i_batch]  # (ALL_VIEW,3,H,W)

        # sampling view
        if view_sampling and opt.subsampled_views > 0:
            NV = opt.subsampled_views if view_num is None else view_num
            view_inds = np.random.choice(ALL_VIEW, NV)
            focal = focal.repeat(NV)
            c2w = c2w[view_inds, :, :]
            rgb = rgb[view_inds, :, :, :]
        else:
            focal = focal.repeat(ALL_VIEW)
            NV = ALL_VIEW

            # get origin & direction of all pixels
        compute_radii = opt.rendering_type == 'mip-nerf'
        cam_rays = get_rays_batch(opt.H, opt.W, focal, c2w, compute_radii=compute_radii)  # (2 or 3,NV,H,W,3)

        # sampling [H,W] indices
        NM = 3 if compute_radii else 2

        assert cam_rays.size(0) == NM and cam_rays.size(1) == NV and cam_rays.size(4) == 3

        assert rgb.size(0) == NV and rgb.size(1) == 3
        cam_rays = cam_rays.permute(0, 1, 4, 2, 3)  # (2,NV,3,H,W)
        cam_rays = cam_rays.reshape(NM, NV, 3, -1)  # (2,NV,3,H*W)
        rgb = rgb.reshape(NV, 3, -1)  # (NV,3,H*W)

        if pixel_sampling and opt.subsampled_pixels > 0:
            if 'bbox' in model_input.keys():
                pass
            else:
                pix_inds = np.random.choice(opt.H * opt.W, opt.subsampled_pixels)

            cam_rays = cam_rays[:, :, :, pix_inds].permute(0, 1, 3, 2)  # (2,NV,NP,3)
            rgb = rgb[:, :, pix_inds].permute(0, 2, 1)  # (NV,NP,3)
        else:
            cam_rays = cam_rays.permute(0, 1, 3, 2)  # (2,NV,NP,3)
            rgb = rgb.permute(0, 2, 1)  # (NV,NP,3)

        all_scene_rays.append(cam_rays)
        all_scene_rgb.append(rgb)

    all_scene_rgb = torch.stack(all_scene_rgb, 0).reshape(bsz, -1, 3)  # (B*NV*NP,3)
    all_scene_rays = torch.stack(all_scene_rays, 1).reshape(NM, -1, 3)  # (3,B*NV*NP,3)
    if opt.rendering_type == 'mip-nerf':
        t_vals, (coords, coords_covs) = sample_along_rays_mip(all_scene_rays, opt)
    else:
        t_vals, coords = sample_along_rays(all_scene_rays, opt)

    coords = coords.reshape(bsz, -1, 3)

    # Model input
    model_input['coords'] = coords
    model_input['rays_d'] = all_scene_rays[1]
    model_input['t_vals'] = t_vals
    del model_input['focal']
    del model_input['c2w']

    # GT
    gt['img'] = all_scene_rgb
    return model_input, gt



def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd):
    eps = 1e-10

    dists = torch.cat(
        [
            t_vals[..., 1:] - t_vals[..., :-1],
            torch.ones(t_vals[..., :1].shape, device=t_vals.device) * 1e10,
        ],
        dim=-1,
    )
    dists = dists * torch.norm(dirs[..., None, :], dim=-1)
    alpha = 1.0 - torch.exp(-density[..., 0] * dists)
    accum_prod = torch.cat(
        [
            torch.ones_like(alpha[..., :1]),
            torch.cumprod(1.0 - alpha[..., :-1] + eps, dim=-1),
        ],
        dim=-1,
    )

    weights = alpha * accum_prod

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    depth = (weights * t_vals).sum(dim=-1)
    acc = weights.sum(dim=-1)
    inv_eps = 1 / eps

    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc[..., None])

    return comp_rgb, acc, depth, weights

def nerf_volume_rendering(prediction, opt, out_type='rgb'):
    pred_rgb, pred_density = prediction['model_out'][..., :3], prediction['model_out'][..., -1:]

    bsz = pred_rgb.shape[0]
    # rgb activation
    pred_rgb = pred_rgb.reshape(-1, opt.num_samples_per_ray + 1, 3)
    if opt.rgb_activation == 'sigmoid':
        pred_rgb = torch.sigmoid(pred_rgb)
    elif opt.rgb_activation == 'relu':
        pred_rgb = F.relu(pred_rgb)
    elif 'sine' in opt.rgb_activation:
        w0 = float(opt.rgb_activation.split('sine')[-1])
        pred_rgb = torch.sin(w0 * pred_rgb)
    elif opt.rgb_activation == 'no_use':
        pass
    else:
        raise Exception("check rgb activation")

    # density activation
    MAX_DENSITY = 1
    pred_density = pred_density.reshape(-1, opt.num_samples_per_ray + 1, 1)
    if opt.density_activation == 'elu':
        pred_density = F.elu(pred_density, alpha=0.1) + 0.1
        pred_density = torch.clip(pred_density, 0, MAX_DENSITY)
    elif opt.density_activation == 'relu':
        pred_density = F.relu(pred_density)
    elif opt.density_activation == 'leakyrelu':
        pred_density = F.leaky_relu(pred_density) + 0.1
    elif opt.density_activation == 'shift1':
        pred_density = torch.clip(pred_density + 1.0, 0, MAX_DENSITY)
    elif opt.density_activation == 'shift':
        pred_density = pred_density + 0.5
        pred_density = torch.clip(pred_density, 0, MAX_DENSITY)
    elif opt.density_activation == 'sine5+shift0.9':
        pred_density = torch.sin(5.0 * pred_density)
        pred_density = torch.clip(pred_density + 0.9, 0, MAX_DENSITY)
    elif opt.density_activation == 'sine5+shift1':
        pred_density = torch.sin(5.0 * pred_density)
        pred_density = torch.clip(pred_density + 1, 0, MAX_DENSITY)
    elif opt.density_activation == 'sine5+shift1.1':
        pred_density = torch.sin(5.0 * pred_density)
        pred_density = torch.clip(pred_density + 1.1, 0, MAX_DENSITY)
    elif opt.density_activation == 'sine5+shift0.5+elu':
        pred_density = torch.sin(5.0 * pred_density) + 0.5
        pred_density = F.elu(pred_density, alpha=0.1) + 0.1
        pred_density = torch.clip(pred_density, 0, MAX_DENSITY)
    elif opt.density_activation == 'sine5+shift1+elu':
        pred_density = torch.sin(5.0 * pred_density) + 1.0
        pred_density = F.elu(pred_density, alpha=0.1) + 0.1
        pred_density = torch.clip(pred_density, 0, MAX_DENSITY)
    elif 'elu+scale' in opt.density_activation:
        scale = float(opt.density_activation.split('elu+scale')[-1])
        pred_density = pred_density * scale
        pred_density = F.elu(pred_density, alpha=0.1) + 0.1
        pred_density = torch.clip(pred_density, 0, MAX_DENSITY)
    elif opt.density_activation == 'no_use':
        pass
    else:
        raise Exception("check density activation")

    t_vals, rays_d = prediction['model_in']['t_vals'], prediction['model_in']['rays_d']
    # if opt.rendering_type == 'functa':
    #     color, acc, depth, weight = volumetric_rendering_functa(pred_rgb, pred_density, t_vals, rays_d, opt.white_bkgd)
    # elif opt.rendering_type == 'mip-nerf':
    #     color, acc, depth, weight = volumetric_rendering_mip(pred_rgb, pred_density, t_vals, rays_d, opt.white_bkgd)
    # else:
    #     color, acc, depth, weight = volumetric_rendering(pred_rgb, pred_density, t_vals, rays_d, opt.white_bkgd)

    color, acc, depth, weight = volumetric_rendering(pred_rgb, pred_density, t_vals, rays_d, opt.white_bkgd)

    # reshape
    color = color.reshape(bsz, -1, 3)
    depth = depth.reshape(bsz, -1)
    acc = acc.reshape(bsz, -1)
    if out_type == 'all':
        prediction['model_out'] = {
            'rgb': color,
            'depth': depth,
            'acc': acc,
        }
    else:
        prediction['model_out'] = color
    return prediction