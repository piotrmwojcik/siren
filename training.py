'''Implements a generic training loop.
'''

import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt


def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, device, writer, save_img = False, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False,
          loss_schedules=None):
    psnrs = []
    time1 = time.time()

    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    # copy settings from Raissi et al. (2019) and here
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')

    model_dir_split = model_dir.split('/')
    log_dir = "/".join(model_dir_split[:-2])
    model_name = model_dir_split[-2]

    if os.path.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
        if val == 'y':
            shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    # summaries_dir = os.path.join(model_dir, 'summaries')
    # utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    # writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            # if not epoch % epochs_til_checkpoint and epoch:
            # torch.save(model.state_dict(),
            #            os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
            # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
            #            np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                labels = gt['img'].squeeze()  # shape: [200000]

                inside_indices = (labels == 1).nonzero(as_tuple=True)[0]
                outside_indices = (labels == 0).nonzero(as_tuple=True)[0]

                num_inside_samples = 500
                num_outside_samples = 15884

                inside_sampled_indices = inside_indices[torch.randperm(inside_indices.size(0))[:num_inside_samples]]
                outside_sampled_indices = outside_indices[torch.randperm(outside_indices.size(0))[:num_outside_samples]]

                sampled_indices = torch.cat([inside_sampled_indices, outside_sampled_indices])
                sampled_indices = sampled_indices[torch.randperm(16384)]

                gt['img'] = gt['img'][0][sampled_indices].unsqueeze(0)
                model_input['coords'] = model_input['coords'][0][sampled_indices].permute(1, 0).view(3, 128,128).unsqueeze(0)

                szum = torch.randn_like(model_input['coords'].squeeze()) * 0.01
                model_input['coords'] += szum

                # unique_values, counts = np.unique(gt['img'], return_counts=True)
                # for value, count in zip(unique_values, counts):
                #     print(f"Value: {value}, Count: {count}")



                model_input = {key: value.to(device) for key, value in model_input.items()}
                gt = {key: value.to(device) for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                if use_lbfgs:
                    def closure():
                        optim.zero_grad()
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt)
                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            train_loss += loss.mean()
                        train_loss.backward()
                        return train_loss

                    optim.step(closure)

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        # writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    # writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                # train_losses.append(train_loss.item())
                # writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    psnrs.append(utils.calculate_psnr((64, 64, 64), model_output, gt))

                if not use_lbfgs:
                    optim.zero_grad()
                    train_loss.backward()

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                    optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (
                    epoch, train_loss, time.time() - start_time))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:
                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model.train()

                total_steps += 1

        if save_img:
            path = os.path.join(log_dir,'comparisions')
            os.makedirs(path, exist_ok=True)

            gt = gt['img'].squeeze(0).reshape(128,128)
            res = model_output['model_out'].detach().squeeze(0).round().reshape(128,128)
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(gt, cmap='gray')
            axes[0].set_title('GT')
            axes[0].axis('off')

            axes[1].imshow(res, cmap='gray')
            axes[1].set_title('Model Output')
            axes[1].axis('off')

            plt.tight_layout()
            plt.savefig(f"{path}/comparison_{model_name}.png", dpi=300, bbox_inches='tight')

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
    time2 = time.time()

    print(f"Training took {time2-time1} seconds")
    return psnrs


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)
