import torch

print(torch.backends.mps.is_available())

x = torch.randn(3,3,device = "mps")
print(x.device)
