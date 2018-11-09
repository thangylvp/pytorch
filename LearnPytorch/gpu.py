from __future__ import print_function
import torch

xx = torch.tensor([5, 4])

if torch.cuda.is_available():
    print("Found cuda")
    device = torch.device("cuda")
    tmp = xx.to(device)
    print('tmp', type(tmp), tmp)
    
