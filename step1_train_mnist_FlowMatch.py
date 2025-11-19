"""
ref: https://github.com/LTH14/JiT/blob/main/denoiser.py
"""
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import MNISTFlowModel    
from utils_data import create_mnist_dataloaders
import os
import math
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--ckpt', type=str, help='define checkpoint path', default=None)
    parser.add_argument('--n_samples', type=int, help='define sampling amounts after every epoch trained', default=36)
    parser.add_argument('--model_base_dim', type=int, help='base dim of Unet', default=64)
    parser.add_argument('--timesteps', type=int, help='sampling steps of DDPM', default=1000)
    parser.add_argument('--P_mean', type=float, help='mean of P', default=-0.8)
    parser.add_argument('--P_std', type=float, help='std of P', default=0.8)
    parser.add_argument('--log_freq', type=int, help='training log message printing frequence', default=5)
    args = parser.parse_args()
    return args


def main(args):
    device = "cuda" if torch.cuda.is_available()  else "cpu"
    train_dataloader, test_dataloader = create_mnist_dataloaders(batch_size=args.batch_size, image_size=28)
    model = MNISTFlowModel(timesteps=args.timesteps,
                           image_size=28,
                           in_channels=1,
                           base_dim=args.model_base_dim,
                           dim_mults=[2, 4],
                           P_mean=args.P_mean,
                           P_std=args.P_std,
                           device=device).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(optimizer, args.lr, total_steps=args.epochs * len(train_dataloader), pct_start=0.25, anneal_strategy='cos')

    # load checkpoint
    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt["model"])
        print("Load checkpoint from {}".format(args.ckpt))

    global_steps = 0
    losses = []
    for i in range(args.epochs):
        model.train()
        for j, (image, target) in enumerate(train_dataloader):
            image = image.to(device)
            loss = model(image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_steps += 1
            losses.append(loss.detach().cpu().item())
        if i % args.log_freq == 0:
            print("Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(i + 1, args.epochs, j, len(train_dataloader),loss.detach().cpu().item(),scheduler.get_last_lr()[0]))
        ckpt = {"model": model.state_dict()}
        os.makedirs("results", exist_ok=True)
        torch.save(ckpt, "results/FM_x_pred_steps_{:0>8}.pt".format(global_steps))

        model.eval()
        samples = model.generate(args.n_samples)
        save_image(samples, "results/FM_x_pred_steps_{:0>8}.png".format(global_steps), nrow=int(math.sqrt(args.n_samples)))

        plt.figure()
        plt.plot(losses)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.savefig("results/FM_x_pred_loss_curve.png")
        plt.close()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)