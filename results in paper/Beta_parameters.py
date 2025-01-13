import torch

def cal_parameter_from_beta(Par, temp_K):
    temp_K = temp_K + 273.15
    # Chaboche Parameters calculations
    E = Par[0] * (1 - Par[1] * torch.exp(temp_K / Par[2]))
    A = (Par[11] * torch.exp(temp_K / Par[12]))
    n = Par[40] + Par[13] / (1 + torch.exp(Par[14] * (temp_K - Par[15])))
    C1 = Par[16] * (1 - Par[17] * torch.exp(temp_K / Par[18]))
    gam1 = Par[37] + Par[19] * torch.exp(temp_K / Par[20])
    kX1 = (Par[21] / (1 + torch.exp(-Par[22] * (temp_K - Par[38]))))
    C2 = Par[23] * (1 - Par[24] * torch.exp(temp_K / Par[25]))
    gam2 = Par[41] + Par[26] * torch.exp(temp_K / Par[27])
    kX2 = (Par[42] / (1 + torch.exp(-Par[28] * (temp_K - Par[29]))))
    Q1 = Par[30] * (1 - Par[31] * torch.exp(temp_K / Par[32]))
    b1 = Par[43] + (Par[33]) / (1 + torch.exp(-Par[34] * (temp_K - Par[6])))
    KR1 = Par[36] + (Par[35]) / (1 + torch.exp(-Par[4] * (temp_K - Par[5])))
    R0 = Par[3] * Q1

    return E, A, n, C1, gam1, kX1, C2, gam2, kX2, Q1, b1, KR1, R0

def cal_parameter_from_beta_tensor(Par, temp_K):
    temp_K = temp_K + 273.15
    temp_K = temp_K.repeat(Par.shape[0], 1)
    # Chaboche Parameters calculations
    E = Par[:, 0] * (1 - Par[:, 1] * torch.exp(temp_K / Par[:, 2]))
    A = (Par[:, 11] * torch.exp(temp_K / Par[:, 12]))
    n = Par[40] + Par[13] / (1 + torch.exp(Par[14] * (temp_K - Par[:, 15])))
    C1 = Par[:, 16] * (1 - Par[:, 17] * torch.exp(temp_K / Par[:, 18]))
    gam1 = Par[:, 37] + Par[:, 19] * torch.exp(temp_K / Par[:, 20])
    kX1 = (Par[:, 21] / (1 + torch.exp(-Par[:, 22] * (temp_K - Par[:, 38]))))
    C2 = Par[:, 23] * (1 - Par[:, 24] * torch.exp(temp_K / Par[:, 25]))
    gam2 = Par[:, 41] + Par[:, 26] * torch.exp(temp_K / Par[:, 27])
    kX2 = (Par[:, 42] / (1 + torch.exp(-Par[:, 28] * (temp_K - Par[:, 29]))))
    Q1 = Par[:, 30] * (1 - Par[:, 31] * torch.exp(temp_K / Par[:, 32]))
    b1 = Par[:, 43] + (Par[:, 33]) / (1 + torch.exp(-Par[:, 34] * (temp_K - Par[:, 6])))
    KR1 = (Par[:, 36] + (Par[:, 35]) / (1 + torch.exp(-Par[:, 4] * (temp_K - Par[:, 5]))))
    R0 = Par[:, 3] * Q1

    return E, A, n, C1, gam1, kX1, C2, gam2, kX2, Q1, b1, KR1, R0

def cal_parameter_from_beta_39(Par, temp_K):
    temp_K = temp_K + 273.15
    # Chaboche Parameters calculations
    E = Par[0] * (1 - Par[1] * torch.exp(temp_K / Par[2]))
    A = (Par[11] * torch.exp(temp_K / Par[12]))
    n = Par[40] + Par[13] / (1 + torch.exp(Par[14] * (temp_K - Par[15])))
    C1 = Par[16] * (1 - Par[17] * torch.exp(temp_K / Par[18]))
    gam1 = Par[37] + Par[19] * torch.exp(temp_K / Par[20])
    kX1 = (Par[21] / (1 + torch.exp(-Par[22] * (temp_K - Par[38]))))
    C2 = Par[23] * (1 - Par[24] * torch.exp(temp_K / Par[25]))
    gam2 = Par[41] + Par[26] * torch.exp(temp_K / Par[27])
    kX2 = (Par[42] / (1 + torch.exp(-Par[28] * (temp_K - Par[29]))))
    Q1 = Par[30] * (1 - Par[31] * torch.exp(temp_K / Par[32]))
    b1 = Par[43] + (Par[33]) / (1 + torch.exp(-Par[34] * (temp_K - Par[6])))
    KR1 = Par[36] + (Par[35]) / (1 + torch.exp(-Par[4] * (temp_K - Par[5])))
    R0 = Par[3] * Q1

    return E, A, n, C1, gam1, kX1, C2, gam2, kX2, Q1, b1, KR1, R0