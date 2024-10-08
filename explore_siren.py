import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

import time

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def print_layer_weights(self):
        for idx, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):  # Check if the layer is Linear
                print(f"Layer {idx} weights:")
                print(layer.weight.data.shape)
                if layer.bias is not None:
                    print(f"Layer {idx} bias:")
                    print(layer.bias.data.shape)
            elif hasattr(layer.linear, 'weight'):  # For custom layers like SineLayer
                print(f"Layer {idx} weights:")
                print(layer.linear.weight.data.shape)
                if hasattr(layer.linear, 'bias') and layer.linear.bias is not None:
                    print(f"Layer {idx} bias:")
                    print(layer.linear.bias.data.shape)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img


def get_image_tensor(image_path, sidelength):
    # Load the image from the specified file path (ensure it's in RGB mode)
    img = Image.open(image_path).convert('RGB')

    # Define the transformation pipeline
    transform = Compose([
        Resize((sidelength, sidelength)),  # Resize to the specified side length
        ToTensor(),                        # Convert the image to a tensor
        #Normalize(mean=torch.Tensor([0.5, 0.5, 0.5]), std=torch.Tensor([0.5, 0.5, 0.5]))  # Normalize the tensor
    ])

    # Apply the transformations
    img = transform(img)
    return img

class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        #img = get_cameraman_tensor(sidelength)
        img = get_image_tensor('data/red_car.png', sidelength)
        self.pixels = img.permute(1, 2, 0).view(128*128, 3)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        print(self.coords.shape, self.pixels.shape)
        return self.coords, self.pixels


cameraman = ImageFitting(128)
dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

img_siren1 = Siren(in_features=2, out_features=3, hidden_features=256,
                  hidden_layers=3, outermost_linear=True)
img_siren2 = Siren(in_features=2, out_features=3, hidden_features=256,
                  hidden_layers=3, outermost_linear=True)
#img_siren.print_layer_weights()

img_siren1#.cuda()
img_siren2#.cuda()

total_steps = 500  # Since the whole image is our dataset, this just means 500 gradient descent steps.
steps_til_summary = 10

optim = torch.optim.Adam(lr=1e-4, params=(list(img_siren1.parameters()) + list(img_siren2.parameters())))


def generate_mlp_from_weights(weights):
    mlp = Siren(in_features=2, out_features=3, hidden_features=256,
                hidden_layers=3, outermost_linear=True)
    state_dict = mlp.state_dict()
    weight_names = list(state_dict.keys())
    for layer in weight_names:
        val = state_dict[layer]
        num_params = np.product(list(val.shape))
        w = weights[:num_params]
        w = w.view(*val.shape)
        state_dict[layer] = w
        weights = weights[num_params:]
    assert len(weights) == 0, f"len(weights) = {len(weights)}"
    mlp.load_state_dict(state_dict)
    return mlp



model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input, ground_truth

for step in range(total_steps):
    model_output1, coords1 = img_siren1(model_input)
    model_output2, coords2 = img_siren2(model_input)
    model_output = torch.cat([model_output1, model_output2], dim=0)
    loss = ((model_output - ground_truth.repeat(2, 1, 1)) ** 2).mean()

    if not step % steps_til_summary:
        print("Step %d, Total loss %0.6f" % (step, loss))
        img_grad = gradient(model_output1, coords1)
        img_laplacian = laplace(model_output, coords1)

        fig, axes = plt.subplots(1, 4, figsize=(18, 6))
        axes[0].imshow(model_output1.cpu().view(128, 128, 3).detach().numpy())
        axes[1].imshow(ground_truth[0].cpu().view(128, 128, 3).detach().numpy())
        axes[2].imshow(img_grad.norm(dim=-1).cpu().view(128, 128).detach().numpy())
        axes[3].imshow(img_laplacian.cpu().view(128, 128).detach().numpy())
        save_path = os.path.join('test_output', f"step_{step}.png")
        plt.savefig(save_path)

    optim.zero_grad()
    loss.backward()
    optim.step()

z = []
state_dict = img_siren1.state_dict()
layers = []
layer_names = []
input = []
for l in state_dict:
    st_shape = state_dict[l].shape
    layers.append(np.prod(st_shape))
    layer_names.append(l)
    input.append(state_dict[l].flatten())
input = torch.hstack(input).cuda()
siren_example = generate_mlp_from_weights(input)#.cuda()
model_input = get_mgrid(128, 2).unsqueeze(0)
img, _ = siren_example(model_input)
print()
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
axes[0].imshow(img.cpu().view(128, 128, 3).detach().numpy())
save_path = os.path.join('test_output', f"final.png")
plt.savefig(save_path)