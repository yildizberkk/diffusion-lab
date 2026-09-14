from ddpm.schedule import Schedule
from ddpm.model import UNet
from ddpm.data import get_dataset
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import time
import copy


def main():

    # Arguments
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, default="mnist", help="Name of the dataset, currently it is either 'mnist' or 'fashion'")
    p.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    p.add_argument("--batch-size", type=int, default=128, help="Size of the batch")
    p.add_argument("--lr", type=float, default=2e-4, help="The learning rate parameter")
    p.add_argument("--base", type=int, default=64, help="UNet base channel width")
    p.add_argument("--T", type=int, default=1000, help="Number of timesteps to iterate")
    p.add_argument("--device", type=str, default="mps", help="The name of the device to run")
    p.add_argument("--out", type=str, default="checkpoints", help="Output folder name for the checkpoints to save")
    p.add_argument("--seed", type=int, default=0, help="The seed to choose")
    p.add_argument("--attention", default=True, action=argparse.BooleanOptionalAction, help="Attention activation")
    p.add_argument("--ema-decay", type=float, default=0.999, help="The decay parameter of the EMA")
    p.add_argument("--schedule", type=str, default="linear", help="The kind of the scheduler. Either linear (default) or cosine")


    args = p.parse_args()
    torch.manual_seed(args.seed)

    

    # Build
    schedule = Schedule(T=args.T, device=args.device, kind=args.schedule)
    model = UNet(base=args.base, attention=args.attention).to(args.device)
    ema_model = copy.deepcopy(model)
    loader = DataLoader(get_dataset(name=args.dataset), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses = []

    for p in ema_model.parameters():
        p.requires_grad_(False)

        
    os.makedirs(args.out, exist_ok=True)

    # Print the initializations
    print(f"device={args.device}  dataset={args.dataset}  epochs={args.epochs}  "
          f"batch={args.batch_size}  lr={args.lr}  base={args.base}  "
          f"T={args.T}  seed={args.seed}  "
          f"attention={args.attention}  "
          f"ema_decay={args.ema_decay}  "
          f"schedule={args.schedule}", flush=True)
    print(f"data:  {len(loader.dataset):,} images, {len(loader)} batches/epoch", flush=True)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)


    # Loop
    for epoch in range(args.epochs):

        epoch_start = time.time()

        for i, (images, _) in enumerate(loader):
            # move images to the device
            x_0 = images.to(args.device)

            # random timesteps, one per image
            t   = torch.randint(0, schedule.T, (x_0.shape[0],), device=args.device)

            # q_sample
            x_t, eps = schedule.q_sample(x_0, t)

            # the model's guess
            pred = model(x_t, t)

            # mse against eps
            loss = F.mse_loss(pred, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                    p_ema.mul_(args.ema_decay).add_(p, alpha=1 - args.ema_decay)

            losses.append(loss.item())

            if i % 50 == 0:
                print(f"epoch {epoch} batch {i}  loss {loss.item():.4f}", flush=True)

        # Primt the epoch with loss
        epoch_loss = sum(losses[-len(loader):]) / len(loader)
        print(f"epoch {epoch:3d} | loss {epoch_loss:.4f} | {time.time() - epoch_start:.0f}s",
              flush=True)

        # A checkpoint for every 5th epoch
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "losses": losses,
            "epoch": epoch,
            "config": vars(args),
            "ema": ema_model.state_dict()
        }

        torch.save(ckpt, f"{args.out}/ckpt_last.pt")

        if epoch % 5 == 0:
            torch.save(ckpt, f"{args.out}/ckpt_{epoch}.pt")



if __name__ == "__main__":
    main()
