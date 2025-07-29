'''
@Author  ：zqp
@Date    ：2024/3/5 14:32
'''
import os

import numpy as np
from math import log10
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import Matern

from datetime import datetime
timestamp = datetime.now().strftime('%Y-%b-%d_%H-%M')
#%%
def SNR(original_data, noisy_data):
    original_data = np.asarray(original_data, dtype=np.float64)
    noisy_data = np.asarray(noisy_data, dtype=np.float64)

    signal_power = np.mean(original_data ** 2)
    noise_power = np.mean((original_data - noisy_data) ** 2)

    snr = signal_power / noise_power
    snr_db = 10 * np.log10(snr)
    return snr_db


def delete_file_if_exists(file_path):
    # Check if file exists
    if os.path.exists(file_path):
        # Delete the file
        os.remove(file_path)
        print(f"The file {file_path} has been deleted.")
    else:
        print(f"The file {file_path} does not exist.")


def RBF_prior(height, width, gamma, d):
    """
    self writed
    0<=x<=1
    0<=y<=1
    Phi_ij = ||(x_i,y_i)-(x_j,y_j)||_2
    K(i,j) = gamma*exp(-Phi_ij_2/d)
    """
    xs = np.linspace(0, 1, width + 1)
    delta_x = (xs[1] - xs[0]) / 2
    xs = xs[:-1] + delta_x

    ys = np.linspace(0, 1, height + 1)
    ys = ys[:-1] + delta_x

    [X, Y] = np.meshgrid(xs, ys)
    XYpoint = np.c_[X.flatten(), Y.flatten()]

    n = height * width
    dist_upper = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist_upper[i, j] = np.linalg.norm(XYpoint[i, :] - XYpoint[j, :])
    distMatrix = dist_upper + dist_upper.T

    K = gamma * np.exp(-distMatrix / d)
    return K

def PSNR(ground_truth, predict):
    """
    """
    ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min())
    predict = (predict - predict.min()) / (predict.max() - predict.min())
    mse = np.mean((ground_truth - predict) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return -1
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel / np.sqrt(mse))
    return np.round(psnr,2)



def Matern_prior(img_size=3,length_scale=0.2,nu=0.5):
    xs = np.linspace(0, 1, img_size + 1)
    delta_x = (xs[1] - xs[0]) / 2
    xs = xs[:-1] + delta_x

    ys = np.linspace(0, 1, img_size + 1)
    ys = ys[:-1] + delta_x

    [X, Y] = np.meshgrid(xs, ys)
    XYpoint = np.c_[X.flatten(), Y.flatten()]

    matern_kernel = Matern(length_scale=length_scale, nu=nu)
    T = matern_kernel(XYpoint)
    return  T

# def total_variation(img):
#     """
#     Compute total variation statistics on the current batch using TensorFlow.
#     """
#     # 检查输入是否为 2D 张量
#     if img.ndim != 2:
#         raise RuntimeError(f"Expected input `img` to be a 2D tensor, but got {img.shape}")
#
#     # 计算行梯度的绝对值
#     diff1 = img[1:, :] - img[:-1, :]
#     abs_diff1 = tf.abs(diff1)
#
#     # 计算列梯度的绝对值
#     diff2 = img[:, 1:] - img[:, :-1]
#     abs_diff2 = tf.abs(diff2)
#
#     # 合并行和列的 TV 值
#     tv_value = tf.reduce_sum(abs_diff1) + tf.reduce_sum(abs_diff2)
#
#     return tv_value

# def total_variation(img):
#     """Compute total variation statistics on current batch."""
#     if img.ndim != 2:
#         raise RuntimeError(f"Expected input `img` to be an 2D tensor, but got {img.shape}")
#     diff1 = img[1:, :] - img[:-1, :]
#     diff2 = img[:, 1:] - img[:, :-1]
#     tv_value = np.abs(diff1).sum()+np.abs(diff2).sum()
#     return tv_value


import torch


def total_variation(img: torch.Tensor) -> torch.Tensor:
    """
    Compute total variation statistics on the input 2D tensor using PyTorch.

    Args:
        img (torch.Tensor): A 2D tensor representing the image.

    Returns:
        torch.Tensor: The total variation value.
    """
    if img.ndim != 2:
        raise RuntimeError(f"Expected input `img` to be a 2D tensor, but got {img.shape}")

    diff1 = img[1:, :] - img[:-1, :]  # Vertical differences
    diff2 = img[:, 1:] - img[:, :-1]  # Horizontal differences
    tv_value = diff1.abs().sum() + diff2.abs().sum()
    return tv_value


# import torch
#
# # horizontal and vertical finite defference operators
# def diffh(x):
#     x_diffh = torch.zeros(x.shape)
#     x_diffh[:,:-1] = x[:,1:] - x[:,0:-1]
#     return x_diffh
#
# def diffv(x):
#     x_diffv = torch.zeros(x.shape)
#     x_diffv[:-1, :] = x[1:, :] - x[0:-1,:]
#     return x_diffv.T
#
# # Total variation norm
# def TVnorm(x):
#     y = torch.sum(torch.sqrt(diffh(x)**2 + diffv(x)**2))
#     return y
#
# def Grad_Image(x, device):
#
#     with torch.no_grad():
#
#         x = x.to(device).clone()
#         x_temp = x[1:, :] - x[0:-1,:]
#         dux = torch.cat((x_temp.T,torch.zeros(x_temp.shape[1],1,device=device)),1).to(device)
#         dux = dux.T
#         x_temp = x[:,1:] - x[:,0:-1]
#         duy = torch.cat((x_temp,torch.zeros((x_temp.shape[0],1),device=device)),1).to(device)
#         return  torch.cat((dux,duy),dim=0).to(device)
def total_variation_grad(x):
    """
    计算图像的总变异正则化项的梯度
    """
    grad_x = np.gradient(x, axis=1)
    grad_y = np.gradient(x, axis=0)

    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 避免除以零
    epsilon = 1e-8
    grad_magnitude[grad_magnitude < epsilon] = epsilon

    tv_grad_x = grad_x / grad_magnitude
    tv_grad_y = grad_y / grad_magnitude

    # 计算总变异正则化项的梯度
    grad = np.zeros_like(x)
    grad += np.gradient(tv_grad_x, axis=1)
    grad += np.gradient(tv_grad_y, axis=0)

    return grad


def sample_cumulated_sum(x_sum_pre,x_square_sum_pre,x_i):
    x_sum = x_sum_pre+x_i
    x_square_sum = x_square_sum_pre+x_i**2
    return x_sum,x_square_sum


def show_images(imgs,img_type=None, titles=None, keep_range=True, shape=None, figsize=(8, 8.5),figure_dir='.'):
    """
    Parameters
    ----------
    imgs   [image1(H,W,C), image2,....., imageN],numpy data
    titles
    keep_range
    shape
    figsize

    Returns
    -------

    """
    combined_data = np.array(imgs)

    if titles is None:
        titles = [str(i) for i in range(combined_data.shape[0])]

    # Get the min and max of all images
    if keep_range:
        _min, _max = np.amin(combined_data), np.amax(combined_data)
    else:
        _min, _max = None, None

    if shape is None:
        shape = (1, len(imgs))

    fig, axes = plt.subplots(*shape, figsize=figsize, sharex=True, sharey=True)
    ax = axes.ravel()
    for i, (img, title) in enumerate(zip(imgs, titles)):
        im = ax[i].imshow(img,vmin=_min, vmax=_max)
        ax[i].set_title(title)
        # if i==(len(imgs)-1): #最后一个图加colorbar
        #     divider = make_axes_locatable(ax[i])
        #     cax = divider.append_axes("right", size="10%", pad=0.05)
        #     fig.colorbar(im,ax=ax[i],cax=cax)

    plt.savefig(os.path.join(figure_dir,f'map_{img_type}.png'),dpi=600)
    plt.show()


if __name__ == '__main__':
    T = Matern_prior(img_size=4,length_scale=0.2,nu=0.5)
    plt.figure()
    plt.imshow(T)
    plt.colorbar()
    plt.show()
