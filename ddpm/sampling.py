import torch

# Adding the @torch.no_grad() is crucial here.
@torch.no_grad()
def sample(model, schedule, n_samples, img_shape=(1, 32, 32)):
    model.eval()
    device = next(model.parameters()).device

    # (n, 1, 32, 32) pure noise, on device
    x = torch.randn([n_samples, *img_shape], device=device)

    # 999 down to 0
    for i in reversed(range(schedule.T)):
        # (n,) all equal to i, long dtype, on device
        t = torch.full((n_samples,), i, device=device)
        # (n, 1, 32, 32)
        eps_hat = model(x, t)

        # alphas[i]
        a = schedule.alphas[i]
        # alpha_bars[i]
        a_bar = schedule.alpha_bars[i]
        # betas[i]
        b = schedule.betas[i]

        # fresh noise, but zeros when i == 0
        z = torch.randn_like(x) if i != 0 else torch.zeros_like(x)

        # the reverse step
        x = (1 / a.sqrt()) * (x - ((1 - a) / (1 - a_bar).sqrt()) * eps_hat ) + (b.sqrt() * z)

    return x
