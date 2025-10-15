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


N = timesteps

x_rec = torch.randn_like(image).to(device).requires_grad_(True)

for name,para in model.model.named_parameters():
    para.requires_grad_(False)
A_A_T = A @ A.T
eta = 1
for _ in range(3):
    x_rec = torch.randn_like(image).to(device).requires_grad_(True)
    mse_list = []
    # get_zeta = make_zeta_fn(base_zeta)
    for i in tqdm(range(N-1,0,-1)):
        # line 4 of Algorithm 1: compute E[x_0|x_i]
        pred = model.model(x_rec, torch.tensor([i], device=device, dtype=torch.long))
        s_hat = -pred/(1-alphas_cumprod[i])**0.5

        # line5 of Algorithm 1: compute x0_hat
        x0_hat = (x_rec + (1-alphas_cumprod[i]) * s_hat) / torch.sqrt(alphas_cumprod[i])
        x0_hat = torch.clamp(x0_hat, -1, 1)

        mse = torch.nn.functional.mse_loss(x0_hat, x.unsqueeze(0).unsqueeze(0)).item()
        mse_list.append(mse)
        # 缩放 x0_hat 到 [0,1]
        # x0_hat_scaled = (x0_hat + 1.0) / 2.0
        x0_hat_vec = x0_hat.detach().cpu().numpy().reshape(-1)  # shape: (784,)
        x_vec = x.detach().cpu().numpy().reshape(-1)  # true x


        # === step 2: pdgm 梯度项（计算对数似然的近似）
        sigma_t = ((1 - alphas_cumprod[i])/alphas_cumprod[i])**0.5 #p(x_t|x0)=N(mean,sigma_t**2)
        rt = (sigma_t**2/(1 + sigma_t**2))**0.5 #见原文公式17
        # cov = r_t^2 * (A A^T) + sigma_y^2 I
        cov = rt**2 * A_A_T + sigma_y**2 * torch.eye(A_A_T.shape[0]).to(device)
        x0_hat_vec = x0_hat.reshape(-1,1)
        A_x0 = A @ x0_hat_vec
        # Sigma_y 为 cov
        residual = y - A_x0
        solved = torch.linalg.solve(cov, residual)
        loglike_neg = torch.linalg.norm(0.5 * (residual.T @ solved).squeeze())
        # 计算（12）
        # === Step 3: DDIM 更新 + 梯度项
        z = torch.randn_like(x_rec, device=device)


        alpha_bar = alphas_cumprod[i]
        alpha_bar_prev = alphas_cumprod[i - 1]

        c1 = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        c2 = torch.sqrt(1 - alpha_bar_prev - c1 ** 2)

        x_prev = (
            torch.sqrt(alpha_bar_prev) * x0_hat +
            c2 * pred +
            c1 * z
        )


        # Step 4: gradient of log p(y|xt) w.r.t. xt
        grad = torch.autograd.grad(loglike_neg, x_rec, retain_graph=True)[0]
        grad_norm = torch.norm(grad)
        if grad_norm > 1.0:
            grad = grad / grad_norm
        # zeta = get_zeta(base_zeta,i, grad_norm)
        # zeta_i = zeta
        x_rec = x_prev - torch.sqrt(alphas_cumprod[i]) * grad
        x_rec = torch.clamp(x_rec,-1.0,1.0)
        x_rec = x_rec.detach().requires_grad_(True)
            #%% results
    # 横轴 t 从 1 到 N-1，纵轴是 MSE
    mse_list = mse_list[::-1]
    t_values = list(range(1, len(mse_list) + 1))

    plt.figure(figsize=(6, 4))
    plt.plot(t_values, mse_list, label=f'sigma_y={sigma_y:.3f}')
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