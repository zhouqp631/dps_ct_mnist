import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from utils import total_variation
from model import MNISTDiffusion
from utils_data import create_mnist_dataloaders,create_mnist_6_dataloaders
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

def GradientIm(u, device):
    u_shapex = list(u.shape)
    u_shapex[0] = 1
    z = u[1:, :] - u[:-1, :]
    dux = torch.vstack([z, torch.zeros(u_shapex, device=device)])

    u_shapey = list(u.shape)
    u_shapey[1] = 1
    z = u[:, 1:] - u[:, :-1]
    duy = torch.hstack([z, torch.zeros(u_shapey, device=device)])
    return dux, duy
def DivergenceIm(p1, p2):
    z = p2[:, 1:-1] - p2[:, :-2]
    shape2 = list(p2.shape)
    shape2[1] = 1
    v = torch.hstack([p2[:, 0].reshape(shape2), z, -p2[:, -1].reshape(shape2)])

    shape1 = list(p1.shape)
    shape1[0] = 1
    z = p1[1:-1, :] - p1[:-2, :]
    u = torch.vstack([p1[0, :].reshape(shape1), z, -p1[-1, :].reshape(shape1)])

    return v + u


def chambolle_prox_TV(g1, device, varargin, img_size):
    with torch.no_grad():
        g1 = g1.reshape(img_size, img_size)
        if isinstance(g1, np.ndarray):
            g1 = torch.tensor(g1, device=device)

        g = g1.clone().detach()

        # initialize
        px = torch.zeros(g.shape, device=device)
        py = torch.zeros(g.shape, device=device)
        cont = 1
        k = 0

        # defaults for optional parameters
        tau = 0.249
        tol = 1e-3
        lambd = 1
        maxiter = 20
        verbose = 0

        # read the optional parameters
        for key in varargin.keys():
            if key.upper() == 'LAMBDA':
                lambd = varargin[key]
            elif key.upper() == 'VERBOSE':
                verbose = varargin[key]
            elif key.upper() == 'TOL':
                tol = varargin[key]
            elif key.upper() == 'MAXITER':
                maxiter = varargin[key]
            elif key.upper() == 'TAU':
                tau = varargin[key]
            elif key.upper() == 'DUALVARS':
                M, N = g.shape
                Maux, Naux = varargin[key].shape
                if M != Maux or N != 2 * Naux:
                    print('Wrong size of the dual variables')
                    return
                px = torch.tensor(varargin[key])
                py = px[:, M:]
                px = px[:, 1:M]
            else:
                pass

        ## Main body
        while cont:
            k = k + 1
            # compute Divergence of (px, py)
            divp = DivergenceIm(px, py)
            u = divp - torch.divide(g, lambd).to(device)
            # compute gradient of u
            upx, upy = GradientIm(u, device)

            tmp = torch.sqrt(upx * upx + upy * upy).to(device)
            # upx = upx.reshape(-1,1)
            # upy = upy.reshape(-1,1)
            # tmp = tmp.reshape(-1,1)
            # px = px.reshape(-1,1)
            # py = py.reshape(-1,1)
            # error
            x1 = -upx + tmp * px
            y1 = -upy + tmp * py
            err = torch.sqrt(torch.sum(x1 ** 2 + y1 ** 2))

            # update px and py
            px = torch.divide(px + tau * upx, 1 + tau * tmp).to(device)
            py = torch.divide(py + tau * upy, 1 + tau * tmp).to(device)
            # check of the criterion
            cont = ((k < maxiter) and (err > tol))

        if verbose:
            print(f'\t\t|=====> k = {k}\n')
            print(f'\t\t|=====> err TV = {round(err, 3)}\n')

        return (g - lambd * DivergenceIm(px, py)).reshape(-1, 1)


def max_eigenval(A, At, im_size, tol, max_iter, verbose, device):
    with torch.no_grad():

        # computes the maximum eigen value of the compund operator AtA

        x = torch.normal(mean=0, std=1, size=(im_size, im_size))[None][None].to(device)
        x = x / torch.norm(torch.ravel(x), 2)
        init_val = 1

        for k in range(0, max_iter):
            y = A(x)
            x = At(y)
            val = torch.norm(torch.ravel(x), 2)
            rel_var = torch.abs(val - init_val) / init_val
            if (verbose > 1):
                print('Iter = {}, norm = {}', k, val)

            if (rel_var < tol):
                break

            init_val = val
            x = x / val

        if (verbose > 0):
            print('Norm = {}', val)

        return val


def log_pi(z, x, rho, lambda_tv, img_size=32):
    # z = erf_func(z)
    z0 = z.reshape(img_size, img_size)
    # 将 PyTorch 张量转换为 NumPy 数组
    z0_np = z0.cpu().detach().numpy()
    # 使用 total_variation 函数处理 NumPy 数组
    tvz_np = total_variation(z0_np)
    # 将结果转换为 PyTorch 张量，并将其移动到指定设备（如 GPU）
    tvz = torch.tensor(tvz_np, device=device)
    return -lambda_tv * tvz - (1 / (2 * rho ** 2)) * torch.linalg.norm(x - z) ** 2

def myula_sampling(lambd,  step_size,  x,y, A,  sigma, num_samples,img_true, img_size):
    psnr_values = []
    x_samples = []
    A_T = A.T
    x_rec = x.reshape(-1,1)
    img_true = img_true.cpu().numpy()
    for i in tqdm(range(num_samples)):
        gradf = A_T @ (y-A@x_rec) / sigma ** 2
        proxg = chambolle_prox_TV(x_rec,device,{'lambda':lambd, 'Maxiter':25},img_size)  # 近似
        gradg = (proxg - x_rec) / lambd
        x_rec = x_rec + step_size * (gradf + gradg) + torch.sqrt(torch.tensor(2 * step_size)) * torch.randn_like(x_rec)
        x_rec = torch.clamp(x_rec, -1, 1)
        x_samples.append(x_rec)
        x_samples_array = np.array([x.cpu().numpy() for x in x_samples])
        x_mean = np.mean(x_samples_array, axis=0)
        # 计算当前x的PSNR值并记录下来
        psnr_val = psnr(img_true, x_mean.reshape(img_size, img_size))
        psnr_values.append(psnr_val)
        # psi_diff = log_pi(z0,x,rho, lambda_tv=theta) - log_pi(z1,x,rho, lambda_tv=theta)
        # accepe_prob = min(1.0, torch.exp(psi_diff).item())
        # if torch.rand(1).item() < accepe_prob:
        #     z0 = z1
        # else:
        #     z0 = z0
    return x_samples_array, psnr_values,x_rec

x_noise = x + sigma_y*torch.randn(*x.shape, device=device)
lambd = 0.1
step_size = 1e-6
num_samples = 500
theta = 1
x_array, psnr_list,x_rec = myula_sampling(lambd, step_size, x_noise, y ,A,sigma_y,num_samples,x,image_size)
x_noise = x_noise.cpu().numpy()
x_img = x.cpu().numpy()
x0_img = x_rec.detach().cpu().numpy().reshape(image_size, image_size)
# x0_img = (x0_img+1.0)/2.0
# x0_hat_img = x0_hat.detach().cpu().numpy().squeeze(0).squeeze(0)
# x0_hat_img = (x0_hat_img+1.0)/2.0
print("psnr:",psnr(x_img,x0_img))
print("ssim:",ssim(x_img,x0_img,data_range=1))
# ---Figure----
fig, axes = plt.subplots(3, 1, figsize=(4, 14))
im1 = axes[0].imshow(x_img)
axes[0].set_title(f'ground truth', fontsize=13)
cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

im2 = axes[1].imshow(x0_img)
axes[1].set_title(f'x0\n psnr:{psnr(x_img,x0_img):3f} \n ssim:{ssim(x_img,x0_img,data_range=1):3f} ',fontsize=13)
cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

im3 = axes[2].imshow(x_noise)
axes[2].set_title(f'x0_hat\n psnr:{psnr(x_img,x_noise):3f} \n ssim:{ssim(x_img,x_noise,data_range=1):3f} ',fontsize=13)
cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
fig.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(psnr_list)
plt.title(f'psnr_values')
plt.xlabel(f'Iterations')
plt.ylabel(f'PSNR')
plt.grid(True)
plt.show()