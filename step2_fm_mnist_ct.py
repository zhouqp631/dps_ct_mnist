"""
2025年11月24日
复现论文的Algorithm 1:https://arxiv.org/pdf/2310.04432
"""
#%%
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from model import MNISTFlowModel
from utils_data import create_mnist_dataloaders
from radon_transform import radonTransform
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
from torch.autograd.functional import jacobian

normalize = lambda x: 2*x-1           # [0,1] to [-1,1]
unnomalize = lambda x: (x+1.0)/2.0    # [-1,1] to [0,1]
#%%
image_size = 28
train_dataloader, test_dataloader = create_mnist_dataloaders(batch_size=10, image_size=image_size)
image, target = next(iter(test_dataloader))
x = image[7,0,...].to(device)

angleNum = image_size//2
A = torch.tensor(radonTransform(angleNum, image_size, image_size).copy()).float().to(device) 
y_noise_free = A @ x.reshape(-1, 1)
sigma_y = 0.01 * torch.max(y_noise_free)
y = y_noise_free + sigma_y * torch.randn(*y_noise_free.shape,device=device)

x_fbp = torch.linalg.solve(A.T@A+0.01*torch.eye(A.shape[1]).to(device),A.T@y)
x_fbp = x_fbp.reshape(image_size,image_size)
psnr_fbp = psnr(x.cpu().numpy(),x_fbp.cpu().numpy())
ssim_fbp = ssim(x.cpu().numpy(),x_fbp.cpu().numpy(),data_range=1)
print(f"FBP psnr_fbp: {psnr_fbp}, ssim_fbp: {ssim_fbp}")
# ------Figure------
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
cax2 = axs[1, 0].imshow(x_fbp.cpu().numpy(), extent=(-dx, 180.0 + dx, -dy, y_noise_free.shape[1] + dy), aspect='auto')
axs[1, 0].set_title('FBP')
fig.colorbar(cax2, ax=axs[1, 0], orientation='vertical')
cax3 = axs[1, 1].imshow(sinogram.cpu().numpy() - sinogram_noise_free.cpu().numpy(), extent=(-dx, 180.0 + dx, -dy, y_noise_free.shape[1] + dy), aspect='auto')
axs[1, 1].set_title('Noise')
fig.colorbar(cax3, ax=axs[1, 1], orientation='vertical')
plt.tight_layout()
plt.show()

#%% DPS-Gaussian (Algorithm 1: no line 5&6 )
timesteps = 1000
model = MNISTFlowModel(timesteps=timesteps,
                        image_size=28,
                        in_channels=1,
                        base_dim=64,
                        dim_mults=[2, 4],
                        P_mean=-0.8,
                        P_std=0.8,
                        device=device).to(device)
checkpoint = torch.load(f"results\FM_x_pred_steps_00047838.pt", map_location=device,weights_only=True)
model.load_state_dict(checkpoint["model"],strict=True)

for name,para in model.net.named_parameters():
    para.requires_grad_(False)
#%%
#%%
step_size = 0.005
xt = x_fbp.clone().reshape(1,1,image_size,image_size).to(device) # initialize
zt = xt.clone().requires_grad_(True)
for t in np.arange(0.2, 1.0, step_size): 
    rt2 = (1-t)**2/(t**2+(1-t)**2)
    x1_pred = model.net(zt,torch.tensor([t], device=device, dtype=torch.long))
    x1_pred = torch.clamp(x1_pred, -1.0, 1.0)
    v = (x1_pred-zt)/(1-t)

    # step 6 of Algo-FlowModel-Pokle.png
    precision_matrix = torch.linalg.inv(rt2*A@A.T+sigma_y**2*torch.eye(A.shape[0]).to(device))
    data_error = y-A@x1_pred.reshape(-1,1)
    potential_fun = -0.5*data_error.T@precision_matrix@data_error
    grad_term = torch.autograd.grad(potential_fun, zt, retain_graph=True)[0]
    grad_term = grad_term.detach()

    # step 7
    v_corrected = v + (1-t)/t*grad_term
    
    zt = zt + v_corrected*step_size
    zt = torch.clamp(zt, -1.0, 1.0)
    

#%% results
x_img = unnomalize(x.cpu().numpy())
x0_img = unnomalize(zt.detach().cpu().numpy().squeeze(0).squeeze(0))
x0_hat_img = unnomalize(x1_pred.detach().cpu().numpy().squeeze(0).squeeze(0))
print(f"psnr:{psnr(x_img,x0_img)}, psnr_hat:{psnr(x_img,x0_hat_img)}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im1 = axes[0].imshow(x_img)
axes[0].set_title(f'ground truth',fontsize=13)
cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

im2 = axes[1].imshow(x0_img)
axes[1].set_title(f'reconstruct \n psnr:{psnr(x_img,x0_img):3f} \n ssim:{ssim(x_img,x0_img,data_range=1):3f} ',fontsize=13)
cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

# im3 = axes[2].imshow(x0_hat_img)
# axes[2].set_title(f'x0_hat \n psnr:{psnr(x_img,x0_hat_img):3f} \n ssim:{ssim(x_img,x0_hat_img,data_range=1):3f}',fontsize=13)
# cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
fig.tight_layout()
plt.savefig(f"results_restoration/fm_mnist_ct.png")
plt.show()
# %%
