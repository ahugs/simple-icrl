import torch

def one_minus(x):
    return 1 - x

def safe_log_one_minus(x):
    return torch.log(1-x + 1e-5)

def minus(x):
    return -x