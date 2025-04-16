"""
====DPS 原文：
Diffusion posterior sampling: a new approach to denoising and inpainting.
https://arxiv.org/pdf/2209.14687#page=15.94

====DPS 原文代码：
 https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/guided_diffusion/condition_methods.py
"""
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from model import MNISTDiffusion
from utils_data import create_mnist_dataloaders
from radon_transform import radonTransform
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
#%%
image_size = 28
train_dataloader, test_dataloader = create_mnist_dataloaders(batch_size=1, image_size=image_size)
image, target = next(iter(test_dataloader))
x = image[0,0,...].to(device)
x = (x+1.0)/2.0

angleNum = image_size
# A = torch.tensor(radonTransform(angleNum, image_size, image_size).copy()).float().to(device)   # radon transform=the forward model of computed tomography
A = torch.eye(image_size**2).to(device)   # eye matrix=denoising
y_noise_free = A @ x.reshape(-1, 1)
sigma = 0.01 * torch.max(y_noise_free)
y = y_noise_free + sigma * torch.randn(*y_noise_free.shape,device=device)

# ---Figure----
sinogram_noise_free = y_noise_free.reshape(angleNum, A.shape[0] // angleNum) #.T
sinogram = y.reshape(angleNum, A.shape[0] // angleNum) #.T
dx, dy = 0.5 * 180.0 / max(x.shape), 0.5 / sinogram_noise_free.shape[1]
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
cax0 = axs[0, 0].imshow(x.cpu().numpy())
axs[0, 0].set_title('True image')
fig.colorbar(cax0, ax=axs[0, 0], orientation='vertical')
cax1 = axs[0, 1].imshow(sinogram.cpu().numpy(), extent=(-dx, 180.0 + dx, -dy, y.shape[1] + dy), aspect='auto')
axs[0, 1].set_title('Noisy data')
fig.colorbar(cax1, ax=axs[0, 1], orientation='vertical')
cax2 = axs[1, 0].imshow(sinogram_noise_free.cpu().numpy(), extent=(-dx, 180.0 + dx, -dy, y_noise_free.shape[1] + dy), aspect='auto')
axs[1, 0].set_title('Noise-free data')
fig.colorbar(cax2, ax=axs[1, 0], orientation='vertical')
cax3 = axs[1, 1].imshow(sinogram.cpu().numpy() - sinogram_noise_free.cpu().numpy(), extent=(-dx, 180.0 + dx, -dy, y_noise_free.shape[1] + dy), aspect='auto')
axs[1, 1].set_title('Noise')
fig.colorbar(cax3, ax=axs[1, 1], orientation='vertical')
plt.tight_layout()
plt.show()

#%% DPS-Gaussian (Algorithm 1: no line 5&6 )
timesteps = 1000
model = MNISTDiffusion(timesteps=timesteps,
                        image_size=28,
                        in_channels=1,
                        base_dim=64,
                        dim_mults=[2, 4],
                        device=device).to(device)
checkpoint = torch.load(f"results/steps_00046500.pt", map_location=device,weights_only=True)
model.load_state_dict(checkpoint["model"])

alphas = model.alphas
alphas_cumprod = model.alphas_cumprod
betas = model.betas
#%%
N = timesteps
for zeta in np.linspace(0.2,20,10):
    x_rec = torch.randn_like(image).to(device).requires_grad_(True)
    for name,para in model.model.named_parameters():
        para.requires_grad_(False)

    for i in tqdm(range(N-1,0,-1)):
        # line 4 of Algorithm 1: compute E[x_0|x_i]
        s_hat  =  model.model(x_rec,torch.tensor([i],device=device,dtype=torch.long))
        x0_hat = (x_rec+(1-alphas_cumprod[i])*s_hat)/torch.sqrt(alphas_cumprod[i])
        x0_hat = torch.clamp(x0_hat,-1.0,1.0)


        # line 6 of Algorithm 1: compute p(x_{i-1}|x_i, x_0)
        z = torch.randn_like(x_rec).to(device)
        reverse_std = betas[i] * (1.0 - alphas_cumprod[i-1]) / (1.0 - alphas_cumprod[i])
        x_iminus1 = torch.sqrt(alphas[i])*(1-alphas_cumprod[i-1])/(1-alphas_cumprod[i])*x_rec+ \
        torch.sqrt(alphas_cumprod[i - 1])*betas[i]/(1-alphas_cumprod[i])*x0_hat+reverse_std*z

        # line 7 of Algorithm 1: update x
        norm = torch.linalg.norm(y-A@x0_hat.reshape(-1,1))
        norm_grad = torch.autograd.grad(norm,x_rec)[0]
        zeta_i = zeta/norm.item()
        x_rec = x_iminus1-zeta_i*norm_grad

        x_rec = x_rec.detach().requires_grad_(True)
    #%% results
    x_img = x.cpu().numpy()
    x0_img = x_rec.detach().cpu().numpy().squeeze(0).squeeze(0)
    x0_img = (x0_img+1.0)/2.0
    x0_hat_img = x0_hat.detach().cpu().numpy().squeeze(0).squeeze(0)
    x0_hat_img = (x0_hat_img+1.0)/2.0
    print("psnr:",psnr(x_img,x0_img))
    print("ssim:",ssim(x_img,x0_img,data_range=1))
# ---Figure----
    fig, axes = plt.subplots(3, 1, figsize=(4, 14))
    im1 = axes[0].imshow(x_img)
    axes[0].set_title(f'ground truth', fontsize=13)
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    im2 = axes[1].imshow(x0_img)
    axes[1].set_title(f'x0 \n zeta:{zeta:3f} \n psnr:{psnr(x_img,x0_img):3f} \n ssim:{ssim(x_img,x0_img,data_range=1):3f} ',fontsize=13)
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    im3 = axes[2].imshow(x0_hat_img)
    axes[2].set_title(f'x0_hat \n psnr:{psnr(x_img,x0_hat_img):3f} \n ssim:{ssim(x_img,x0_hat_img,data_range=1):3f}',fontsize=13)
    cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()