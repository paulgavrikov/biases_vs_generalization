import torch


class AddNoise(torch.nn.Module):
  """
  Add diffusion-style noise to the input.
  """
  sqrt_alphas: bool = True
  
  def forward(self, x):

    shape = list(x.shape)
    shape[0] = -1
    for i in range(1, len(shape)):
        shape[i] = 1
    
    alpha = torch.rand(x.size(0), device=x.device).reshape(*shape)
    if self.sqrt_alphas:
      alpha = torch.sqrt(alpha)
    std = torch.sqrt(1 - alpha ** alpha)
    x = x * alpha + std * torch.normal(mean=0., std=1., size=x.shape, device=x.device)
    return x


class DiffusionNoiseModel(torch.nn.Module):
    """
    A wrapper around a model that adds noise to the input.
    """

    def __init__(self, model):
       super().__init__()
       self.noiser = AddNoise()
       self.module = model
       self.is_noise_on = True

    def set_noise(self, is_noise_on):
        self.is_noise_on = is_noise_on

    def forward(self, x):
        if self.is_noise_on:
            x = self.noiser(x)
        x = self.module(x)
        return x
