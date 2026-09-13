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

# This is the DDIM version of the sampling.
# In DDIM, instead of following a Markov Process, they are following another approach and not iterating for T=1000 steps.
# They predict the noise from x_t to x_0 and let' call it \eps_hat. then they derive the x_0 at every step, let call it x_0_hat.
# Then they add the predicted noise back so that pereserving some of x_0_hat and adding some of \eps_hat. In order to arrive to a state x_s, s < t.
# This way they are able to get the same performance of iterating for T=1000 steps in 50 steps instead.
# There is also an \eta parameter which allows to manage the budget of noise, how much of the predicted noise is added back and how much of a fresh noise z we would add instead.
# Note that:
# \eta = 0 -> deterministic DDIM
# \eta = 1 and s = t-1 -> DDPM
# So, we can make choice on DDIM to get the DDPM indeed.
@torch.no_grad()
def ddim_sample(model, schedule, n_samples, img_shape=(1, 32, 32), steps=50):
    model.eval()
    device = next(model.parameters()).device

    # (n, 1, 32, 32) pure noise, on device
    x = torch.randn([n_samples, *img_shape], device=device)

    # the 50 t values from T-1 to 0
    ts = torch.linspace(start=schedule.T-1, end=0, steps=steps).round().long()

    for i in range(steps):
        # We need two timesteps, where we are and where we are going
        t_cur = ts[i]
        t_next = ts[i+1] if i+1 < steps else None

        # (n,) all equal to i, long dtype, on device
        t = torch.full((n_samples,), t_cur, device=device)
        # (n, 1, 32, 32)
        eps_hat = model(x, t)

        # Define a_bar for both t (t_cur) and s (t_next)
        a_bar_t = schedule.alpha_bars[t_cur]
        a_bar_s = schedule.alpha_bars[t_next] if i+1 < steps else torch.tensor(1.0, device=device)

        # Deriving the original image from the noise prediction (as I have explanied above, first iterations will be pretty bad)
        x_0_hat = ((x - (( 1 - a_bar_t).sqrt()) * eps_hat) / a_bar_t.sqrt()).clamp(min = -1, max = 1)

        # Renoise
        x = (a_bar_s.sqrt() * x_0_hat) + ((1 - a_bar_s).sqrt() * eps_hat)

    return x


