# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules, glob

from torch.utils.data import DataLoader
import configargparse
from functools import partial

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=15000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=10000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()

png_files = glob.glob(os.path.join('/data/pwojcik/sh_raw_test/cars_500', "*.png"))

# Iterate over the list of .png files and print full paths
for png_file in png_files:
    full_path = os.path.abspath(png_file)
    file_name = os.path.basename(png_file)
    print(f"Processing file: {full_path}")
    # Here you can add code to process each png_file

    img_dataset = dataio.ImageFile(full_path)
    coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=128, compute_diff='none')
    image_resolution = (128, 128)

    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

    # Define the model.
    if opt.model_type == 'sine' or opt.model_type == 'relu' or opt.model_type == 'tanh' or opt.model_type == 'selu' or opt.model_type == 'elu'\
            or opt.model_type == 'softplus':
        model = modules.SingleBVPNet(type='relu', mode='mlp', out_features=3,
                                     hidden_features=128, sidelength=image_resolution)
    elif opt.model_type == 'rbf' or opt.model_type == 'nerf':
        model = modules.SingleBVPNet(type='relu', mode=opt.model_type, sidelength=image_resolution)
    else:
        raise NotImplementedError
    model.cuda()

    root_path = os.path.join(opt.logging_root, file_name)

    # Define the loss
    loss_fn = partial(loss_functions.image_mse, None)
    summary_fn = partial(utils.write_image_summary, image_resolution)

    training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                   steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                   model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn)
