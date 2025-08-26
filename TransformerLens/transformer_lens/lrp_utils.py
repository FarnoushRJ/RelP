import torch



def stabilize(z):
    return z + ((z == 0.).to(z) + z.sign()) * 1e-6
