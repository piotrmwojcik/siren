'''Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement.
'''

# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import numpy as np
import dataio, meta_modules, utils, training, loss_functions, modules
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=20000)
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


sdf_dataset = dataio.PointCloud(opt.point_cloud_path, on_surface_points=opt.batch_size)
dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

x = sdf_dataset[1]
print(x[1])
# Assuming the data is in dictionary form as given:
data = {
    'coords': x[0]['coords'],
    'sdf': x[1]['sdf'],
    'normals': x[1]['normals']
}

coords = data['coords'].numpy()
sdf = data['sdf'].numpy().flatten()
normals = data['normals'].numpy()

coords = coords[sdf==0]


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

unique_values, counts = np.unique(sdf, return_counts=True)
for value, count in zip(unique_values, counts):
    print(f"Value: {value}, Count: {count}")

# colors = ['cyan' if val == 0 else 'blue' for val in sdf]
scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c="blue", s=5, depthshade=True)

# ax.quiver(coords[:, 0], coords[:, 1], coords[:, 2],
#           normals[:, 0], normals[:, 1], normals[:, 2],
#           length=0.1, color='red', normalize=True)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D SDF Visualization with Normals")

plt.show()


# Define the model.
if opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3)
else:
    model = modules.SingleBVPNet(type=opt.model_type, in_features=3)
model.cuda()

# Define the loss
loss_fn = loss_functions.sdf
summary_fn = utils.write_sdf_summary

root_path = os.path.join(opt.logging_root, opt.experiment_name)

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
               clip_grad=True)
