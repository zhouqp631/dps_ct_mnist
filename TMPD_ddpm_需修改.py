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
from torch.func import vjp
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
# x = (x+1.0)/2.0

angleNum = image_size // 2
A = torch.tensor(radonTransform(angleNum, image_size, image_size).copy()).float().to(device)   # radon transform=the forward model of computed tomography
# A = torch.eye(image_size**2).to(device)   # eye matrix=denoising
y_noise_free = A @ x.reshape(-1, 1)
sigma_y = 0.1 * torch.max(y_noise_free)
y = y_noise_free + sigma_y * torch.randn(*y_noise_free.shape,device=device)

x_fbp = torch
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
checkpoint = torch.load(f"results/mix_steps_00469000.pt", map_location=device,weights_only=True)
model.load_state_dict(checkpoint["model"])
model.eval()
alphas = model.alphas
alphas_cumprod = model.alphas_cumprod
betas = model.betas
#%%
N = timesteps


for name,para in model.model.named_parameters():
    para.requires_grad_(False)

def fun(x_rec):
    x_rec = x_rec.reshape(1,1,28,28)
    pred = model.model(x_rec, torch.tensor([i], device=device, dtype=torch.long))
    s_hat = -pred / (1 - alphas_cumprod[i]) ** 0.5

    # line5 of Algorithm 1: compute x0_hat
    x0_hat = (x_rec + (1 - alphas_cumprod[i]) * s_hat) / torch.sqrt(alphas_cumprod[i])
    x0_hat = torch.clamp(x0_hat, -1, 1)
    res = A @ x0_hat.reshape(-1,1)
    return res
# === 全局配置 ===
mse_list = []
x_rec = torch.randn_like(image).to(device).requires_grad_(True)
for i in tqdm(range(N-1,0,-1)):
    # line 4 of Algorithm 1: compute E[x_0|x_i]
    with torch.no_grad():
        pred = model.model(x_rec, torch.tensor([i], device=device, dtype=torch.long))
    s_hat = -pred/(1-alphas_cumprod[i])**0.5
    del pred

    # line5 of Algorithm 1: compute x0_hat
    x0_hat = (x_rec + (1-alphas_cumprod[i]) * s_hat) / torch.sqrt(alphas_cumprod[i])
    x0_hat = torch.clamp(x0_hat, -1, 1)

    mse = torch.nn.functional.mse_loss(x0_hat, x.unsqueeze(0).unsqueeze(0)).item()
    mse_list.append(mse)
    # line 2 m0_y
    torch.cuda.empty_cache()
    x_clone = x_rec.clone().reshape(-1,1)
    #compute ▽m0*H.T = ▽(H@x0)/xt
    h_x_0, vjp_estimate_h_x_0 = vjp(fun,x_clone)
    #v*▽(H@x0)/xt = H(1)*H*▽x0/xt
    #H(1) = Σj Hij(1204,1)
    #H*▽x0/xt = Σi (ΣjHij*Jjm) (1204,784)
    deltam0_xt = vjp_estimate_h_x_0(A @ torch.ones_like(x0_hat.reshape(-1,1)))[0] # approximate = grad(H@x0)/xt
    term1 = (1 - alphas_cumprod[i]) / torch.sqrt(alphas_cumprod[i]) * A @ deltam0_xt # ratio*H@grad
    term2 = sigma_y**2 * torch.ones(A.size(0),1, device=device)
    C_yy = term1 + term2 #cov
    del deltam0_xt, term1, term2

    residual = y - h_x_0
    ls = (1 - alphas_cumprod[i]) / torch.sqrt(alphas_cumprod[i]) * vjp_estimate_h_x_0(residual/ C_yy)[0] #line5 of Algorithm1
    m0_y = x0_hat + ls.reshape_as(x0_hat)
    # cov
    # deltam0_xt = torch.autograd.grad(x0_hat, x_rec, create_graph=True)[0]
    # jacobian_diag = deltam0_xt.reshape(-1)  # [D]
    # cov = (sigma_n**2 * torch.eye(A.shape[0], device=device) + (((1 - alphas_cumprod[i])
    #                             / torch.sqrt(alphas_cumprod[i])) ** 2) * A @ torch.diag(jacobian_diag ** 2) @ A.T)
    # x0_hat_vec = x0_hat.reshape(-1, 1)
    # A_x0 = A @ x0_hat_vec
    # solved = torch.linalg.solve(cov, y - A_x0)
    # m0_y = x0_hat.reshape(-1,1) + (1 - alphas_cumprod[i]) / torch.sqrt(alphas_cumprod[i]) * torch.diag(jacobian_diag ** 2) @ A.T @ solved
    #line 3-5 ddpm更新xt_bar
    reverse_std = betas[i] * (1.0 - alphas_cumprod[i - 1]) / (1.0 - alphas_cumprod[i])
    z = torch.randn_like(x_rec, device=device)

    x_iminus1 = (torch.sqrt(alphas[i]) * (1 - alphas_cumprod[i - 1]) / (1 - alphas_cumprod[i]) * x_rec +
                 torch.sqrt(alphas_cumprod[i - 1]) * betas[i] / (1 - alphas_cumprod[i]) * m0_y +
                 reverse_std * z)

    del z, ls, residual, m0_y, vjp_estimate_h_x_0, h_x_0
    x_rec = x_iminus1.detach()
    x_rec = torch.clamp(x_rec, -1, 1)
    x_rec = x_rec.requires_grad_(True)
        #%% results
mse_list = mse_list[::-1]
t_values = list(range(1, len(mse_list) + 1))

plt.figure(figsize=(6, 4))
plt.plot(t_values, mse_list, label=f'TMPD_ddpm')
plt.xlabel("Timestep t")
plt.ylabel("MSE(x0_hat, x_true)")
plt.title("MSE between x0_hat and x_true vs t")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
x_img = x.cpu().numpy()
x0_img = x_rec.detach().cpu().numpy().squeeze(0).squeeze(0)
# x0_img = (x0_img+1.0)/2.0
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
axes[1].set_title(f'x0 \n psnr:{psnr(x_img,x0_img):3f} \n ssim:{ssim(x_img,x0_img,data_range=1):3f} ',fontsize=13)
cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

im3 = axes[2].imshow(x0_hat_img)
axes[2].set_title(f'x0_hat \n psnr:{psnr(x_img,x0_hat_img):3f} \n ssim:{ssim(x_img,x0_hat_img,data_range=1):3f}',fontsize=13)
cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
fig.tight_layout()
plt.show()
