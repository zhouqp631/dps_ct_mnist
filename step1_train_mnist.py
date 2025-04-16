"""
https://github.com/bot66/MNISTDiffusion/tree/main?tab=readme-ov-file
"""
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import MNISTDiffusion
from utils_data import create_mnist_dataloaders
import os
import math
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--ckpt', type=str, help='define checkpoint path', default=f'results\steps_00009300.pt')
    parser.add_argument('--n_samples', type=int, help='define sampling amounts after every epoch trained', default=36)
    parser.add_argument('--model_base_dim', type=int, help='base dim of Unet', default=64)
    parser.add_argument('--timesteps', type=int, help='sampling steps of DDPM', default=1000)
    parser.add_argument('--model_ema_steps', type=int, help='ema model evaluation interval', default=10)
    parser.add_argument('--model_ema_decay', type=float, help='ema model decay', default=0.995)
    parser.add_argument('--log_freq', type=int, help='training log message printing frequence', default=10)
    parser.add_argument('--no_clip', action='store_true',
                        help='set to normal sampling method without clip x_0 which could yield unstable samples')
    args = parser.parse_args()

    return args


def main(args):
    device = "cuda" if torch.cuda.is_available()  else "cpu"
    train_dataloader, test_dataloader = create_mnist_dataloaders(batch_size=args.batch_size, image_size=28)
    model = MNISTDiffusion(timesteps=args.timesteps,
                           image_size=28,
                           in_channels=1,
                           base_dim=args.model_base_dim,
                           dim_mults=[2, 4],
                           device=device).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(optimizer, args.lr, total_steps=args.epochs * len(train_dataloader), pct_start=0.25, anneal_strategy='cos')
    loss_fn = nn.MSELoss(reduction='mean')

    # load checkpoint
    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt["model"])
        print("Load checkpoint from {}".format(args.ckpt))

    global_steps = 0
    for i in range(args.epochs):
        model.train()
        for j, (image, target) in enumerate(train_dataloader):
            noise = torch.randn_like(image).to(device)
            image = image.to(device)
            pred = model(image, noise)
            loss = loss_fn(pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            global_steps += 1
            if j % args.log_freq == 0:
                print("Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(i + 1, args.epochs, j, len(train_dataloader),loss.detach().cpu().item(),scheduler.get_last_lr()[0]))
        ckpt = {"model": model.state_dict()}
        os.makedirs("results", exist_ok=True)
        torch.save(ckpt, "results/steps_{:0>8}.pt".format(global_steps))

        model.eval()
        samples = model.sampling(args.n_samples, clipped_reverse_diffusion=not args.no_clip, device=device)
        save_image(samples, "results/steps_{:0>8}.png".format(global_steps), nrow=int(math.sqrt(args.n_samples)))

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)