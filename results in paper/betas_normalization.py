import torch
def normalize_linear_betas(tens, extrema):
    return (tens - extrema[:, 0]) / (extrema[:, 1] - extrema[:, 0])

def denormalize_linear_betas(tens, extrema):
    return tens * (extrema[:, 1] - extrema[:, 0]) + extrema[:, 0]

def normalize_log_betas(tens, extrema_log):
    return (torch.log10(tens) - extrema_log[:, 0]) / (extrema_log[:, 1] - extrema_log[:, 0])

def denormalize_log_betas(tens, extrema_log):
    return 10 ** (tens * (extrema_log[:, 1] - extrema_log[:, 0]) + extrema_log[:, 0])