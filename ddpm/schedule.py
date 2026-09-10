import torch

class Schedule:
    def __init__(self, T=1000, device="cpu"):
        # Number of time steps
        self.T = T

        # Variance (!) of the noise getting added in each step
        self.betas = torch.linspace(1e-4,0.02,self.T, device=device)

        # Complements of noises, how much signal survives in each step
        self.alphas = 1- self.betas

        # Cumulative products of alphas. entry t is the a_1 \cdot \dots \cdot a_t
        # How much of the original image survives after t steps
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Assertions
        assert  torch.allclose( self.alpha_bars[0], torch.tensor(1.0), atol=1e-3 ),  f"alpha_bars should start at ~1, got {self.alpha_bars[0].item()}"
        assert  torch.allclose( self.alpha_bars[-1], torch.tensor(0.0), atol=1e-3 ),  f"alpha_bars should end at ~0, got {self.alpha_bars[-1].item()}"
        assert (self.alpha_bars[1:] < self.alpha_bars[:-1]).all(), "alpha_bars are not strictly decreasing"

    def q_sample(self, x_0, t):
        # 1. draw the noise -> (B, 1, 32, 32)
        eps = torch.randn(x_0.shape, dtype=x_0.dtype, device=x_0.device)

        # 2. look up alpha_bar for each t in the batch  -> (B,)
        ab = self.alpha_bars[t].view(-1, 1, 1, 1)

        # 3. the closed form
        x_t = ab.sqrt() * x_0 + (1-ab).sqrt()* eps

        return x_t, eps


