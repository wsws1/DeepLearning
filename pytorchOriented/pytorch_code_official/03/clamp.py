import torch

a = torch.rand(2, 2) * 10

print(a)
a = a.clamp(2, 5)

print(a)
#更多资源加微信：itit11223344