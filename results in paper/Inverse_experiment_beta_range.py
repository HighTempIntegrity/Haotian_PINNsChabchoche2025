import scipy.io
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io
import time as tm
from Inverse_experiment_beta_range_class import fit
from Inverse_experiment_beta_range_class import cal_beta_limit
import betas_normalization
import time

# Set random seed for reproducibility
seed = 128
torch.manual_seed(seed)
np.random.seed(seed)

# Use date+time for naming
time_now = time.localtime()
time_now = time.strftime('%Y%m%d_%H%M%S', time_now)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_dir = os.path.dirname(__file__)
torch.autograd.set_detect_anomaly(False)

## load file for the normalization of betas
numpy_range = np.loadtxt(current_dir + r'\data\Beta_lilmit_beta_range.txt', dtype=np.float64, delimiter=',')
ranges = torch.tensor(numpy_range, dtype=torch.float64, device=device)

numpy_log_index = np.loadtxt(current_dir + r'\data\log_norm_index_beta_range.txt', dtype=np.float64, delimiter=',')
log_index = torch.tensor(numpy_log_index, dtype=torch.int32)

numpy_linear_index = np.loadtxt(current_dir + r'\data\linear_norm_index_beta_range.txt', dtype=np.float64, delimiter=',')
linear_index = torch.tensor(numpy_linear_index, dtype=torch.int32)

ranges_log = torch.log10(ranges[log_index, :])
ranges_linear = ranges[linear_index, :]

par_true = torch.tensor([
    1.746270443e05, 2.243801741e-03, 245.0381457, 1.993972806e-01, 6.316767047e-02,
    1.204509005e03, 1.117509005e03, 8.510121916, 2.081156209e01, 8.302192727e-03,
    8.388734383e02, 3.663942147e-83, 8.379167369, 2.126480998e01, 9.198109814e-03,
    8.660381500e02, 8.332538299e05, 1.495990142e-01, 8.073018609e03, 2.898935861e03,
    1.181854251e05, 1.355441925e01, 5.053413638e-02, 1.860377041e05, 2.480267858e-01,
    7.125301256e03, 1.073468090e01, 2.823841517e02, 1.571568938e-02, 1.202807068e03,
    6.961872471e02, 1.121622191e-03, 1.898530554e02, 9.1584068397, 2.526706819e-02,
    2.2440000e-03, 5.000000000e-08, 5.731891732e03, 1.117509005e03, 8.833701977,
    9.822449538, 4.787523269e+02, 1.289033897e-01, 2.122267609
], device=device, dtype=torch.float64)
# print("new benchmark in the middle of ranges, with correct bound penalty")
index = 36
print('Using the first ', str(index), ' datasets')
sig_save_stress = torch.tensor(np.loadtxt(current_dir + r'\data\Random_100_dataset_sig_save_stress.txt', dtype=np.float64, delimiter=',')[:index*100, :], device=device, dtype=torch.float64)
t_save_stress = torch.tensor(np.loadtxt(current_dir + r'\data\Random_100_dataset_t_save_stress.txt', dtype=np.float64, delimiter=',')[:index*100, :], device=device, dtype=torch.float64)
sig_save_monotonic = torch.tensor(np.loadtxt(current_dir + r'\data\Random_100_dataset_sig_save_monotonic.txt', dtype=np.float64, delimiter=',')[:index*100], device=device, dtype=torch.float64)
eps_save_monotonic = torch.tensor(np.loadtxt(current_dir + r'\data\Random_100_dataset_eps_save_monotonic.txt', dtype=np.float64, delimiter=',')[:index*100], device=device, dtype=torch.float64)
sig_save_cyclic = torch.tensor(np.loadtxt(current_dir + r'\data\Random_100_dataset_sig_save_cyclic.txt', dtype=np.float64, delimiter=',')[:index*100, :], device=device, dtype=torch.float64)
eps_save_cyclic = torch.tensor(np.loadtxt(current_dir + r'\data\Random_100_dataset_eps_save_cyclic.txt', dtype=np.float64, delimiter=',')[:index*100, :], device=device, dtype=torch.float64)

sig_mean_save = torch.tensor(np.loadtxt(current_dir + r'\data\Random_100_dataset_sig_mean_save.txt', dtype=np.float64, delimiter=',')[:index*100], device=device, dtype=torch.float64)
strain_rate_save_monotonic = torch.tensor(np.loadtxt(current_dir + r'\data\Random_100_dataset_strain_rate_save_monotonic.txt', dtype=np.float64, delimiter=',')[:index*100], device=device, dtype=torch.float64)
strain_rate_save_stress = torch.tensor(np.loadtxt(current_dir + r'\data\Random_100_dataset_strain_rate_save_stress.txt', dtype=np.float64, delimiter=',')[:index*100, :], device=device, dtype=torch.float64)
strain_rate_save_cyclic = torch.tensor(np.loadtxt(current_dir + r'\data\Random_100_dataset_strain_rate_save_cyclic.txt', dtype=np.float64, delimiter=',')[:index*100, :], device=device, dtype=torch.float64)
end_save_monotonic = torch.tensor(np.loadtxt(current_dir + r'\data\Random_100_dataset_end_save_monotonic.txt', dtype=np.float64, delimiter=',')[:index*100], device=device, dtype=torch.float64)
end_save_stress = torch.tensor(np.loadtxt(current_dir + r'\data\Random_100_dataset_end_save_stress.txt', dtype=np.float64, delimiter=',')[:index*100, :], device=device, dtype=torch.float64)
end_save_cyclic = torch.tensor(np.loadtxt(current_dir + r'\data\Random_100_dataset_end_save_cyclic.txt', dtype=np.float64, delimiter=',')[:index*100, :], device=device, dtype=torch.float64)

temp_K = torch.tensor(np.loadtxt(current_dir + r'\data\Random_100_dataset_temperature_save.txt', dtype=np.float64, delimiter=',')[:index*100], device=device, dtype=torch.float64)

# initial guess of model parameters

para0_numpy = np.loadtxt(current_dir + r'\data\Random_100_betas_set.txt', dtype=np.float64, delimiter=',')
para0 = torch.tensor(para0_numpy, dtype=torch.float64, device=device)
par = torch.zeros([100, 44], dtype=torch.float64, device=device)
final_loss = torch.zeros([100], device=device)
training_time = torch.zeros([100], device=device)

beta_upper, beta_lower = cal_beta_limit(par_true, temp_K)
for idx_para in range(100):

    Betas_norm = torch.zeros((para0.shape[1]), device=device, dtype=torch.float64)
    Betas_norm[log_index] = betas_normalization.normalize_log_betas(para0[idx_para, log_index], ranges_log)
    Betas_norm[linear_index] = betas_normalization.normalize_linear_betas(para0[idx_para, linear_index], ranges_linear)
    log_beta = torch.nn.Parameter(Betas_norm).requires_grad_()

    optimizer1 = torch.optim.Adam([log_beta], lr=0.01)

    optimizer2105 = torch.optim.LBFGS([log_beta],
                                      lr=0.5,
                                      max_iter=500,
                                      max_eval=500,
                                      line_search_fn='strong_wolfe')

    optimizer2101 = torch.optim.LBFGS([log_beta],
                                     lr=0.1,
                                     max_iter=1000,
                                     max_eval=1000,
                                     line_search_fn='strong_wolfe')

    optimizer22 = torch.optim.LBFGS([log_beta],
                                    lr=0.01,
                                    max_iter=500,
                                    max_eval=500,
                                    line_search_fn='strong_wolfe')

    start_time = tm.time()

    history0 = fit(num_epochs=10, optimizer=optimizer1, cycle=1, verbose=False, beta=log_beta, beta_upper=beta_upper, beta_lower=beta_lower,
                   temp_K=temp_K, sig_save_stress=sig_save_stress, t_save_stress=t_save_stress,
                   sig_save_monotonic=sig_save_monotonic, eps_save_monotonic=eps_save_monotonic,
                   sig_save_cyclic=sig_save_cyclic, eps_save_cyclic=eps_save_cyclic,
                   strain_rate_save_monotonic=strain_rate_save_monotonic, strain_rate_save_stress=strain_rate_save_stress, strain_rate_save_cyclic=strain_rate_save_cyclic,
                   sig_mean_save=sig_mean_save, end_save_monotonic=end_save_monotonic, end_save_stress=end_save_stress, end_save_cyclic=end_save_cyclic, ranges=ranges, log_index=log_index, linear_index=linear_index, p=2)

    history1 = fit(num_epochs=1, optimizer=optimizer2105, cycle=1, verbose=False, beta=log_beta, beta_upper=beta_upper, beta_lower=beta_lower,
                   temp_K=temp_K, sig_save_stress=sig_save_stress, t_save_stress=t_save_stress,
                   sig_save_monotonic=sig_save_monotonic, eps_save_monotonic=eps_save_monotonic,
                   sig_save_cyclic=sig_save_cyclic, eps_save_cyclic=eps_save_cyclic,
                   strain_rate_save_monotonic=strain_rate_save_monotonic, strain_rate_save_stress=strain_rate_save_stress, strain_rate_save_cyclic=strain_rate_save_cyclic,
                   sig_mean_save=sig_mean_save, end_save_monotonic=end_save_monotonic, end_save_stress=end_save_stress, end_save_cyclic=end_save_cyclic, ranges=ranges, log_index=log_index, linear_index=linear_index, p=2)

    print('finish first p2 training')
    log_beta = torch.nn.Parameter(log_beta.clone().detach()).requires_grad_()


    optimizer2101 = torch.optim.LBFGS([log_beta],
                                      lr=0.1,
                                      max_iter=1000,
                                      max_eval=1000,
                                      line_search_fn='strong_wolfe')


    history3 = fit(num_epochs=1, optimizer=optimizer2101, cycle=2, verbose=False, beta=log_beta, beta_upper=beta_upper, beta_lower=beta_lower,
                   temp_K=temp_K, sig_save_stress=sig_save_stress, t_save_stress=t_save_stress,
                   sig_save_monotonic=sig_save_monotonic, eps_save_monotonic=eps_save_monotonic,
                   sig_save_cyclic=sig_save_cyclic, eps_save_cyclic=eps_save_cyclic,
                   strain_rate_save_monotonic=strain_rate_save_monotonic, strain_rate_save_stress=strain_rate_save_stress, strain_rate_save_cyclic=strain_rate_save_cyclic,
                   sig_mean_save=sig_mean_save, end_save_monotonic=end_save_monotonic, end_save_stress=end_save_stress, end_save_cyclic=end_save_cyclic, ranges=ranges, log_index=log_index, linear_index=linear_index, p=2)

    print('finish last p2 training')


    log_beta = torch.nn.Parameter(log_beta.clone().detach()).requires_grad_()

    optimizer1 = torch.optim.Adam([log_beta], lr=0.01)

    optimizer2105 = torch.optim.LBFGS([log_beta],
                                      lr=0.5,
                                      max_iter=500,
                                      max_eval=500,
                                      line_search_fn='strong_wolfe')

    optimizer2101 = torch.optim.LBFGS([log_beta],
                                      lr=0.1,
                                      max_iter=1000,
                                      max_eval=1000,
                                      line_search_fn='strong_wolfe')

    optimizer22 = torch.optim.LBFGS([log_beta],
                                    lr=0.05,
                                    max_iter=500,
                                    max_eval=500,
                                    line_search_fn='strong_wolfe')

    history5 = fit(num_epochs=1, optimizer=optimizer2105, cycle=1, verbose=False, beta=log_beta, beta_upper=beta_upper, beta_lower=beta_lower,
                   temp_K=temp_K, sig_save_stress=sig_save_stress, t_save_stress=t_save_stress,
                   sig_save_monotonic=sig_save_monotonic, eps_save_monotonic=eps_save_monotonic,
                   sig_save_cyclic=sig_save_cyclic, eps_save_cyclic=eps_save_cyclic,
                   strain_rate_save_monotonic=strain_rate_save_monotonic, strain_rate_save_stress=strain_rate_save_stress, strain_rate_save_cyclic=strain_rate_save_cyclic,
                   sig_mean_save=sig_mean_save, end_save_monotonic=end_save_monotonic, end_save_stress=end_save_stress, end_save_cyclic=end_save_cyclic, ranges=ranges, log_index=log_index, linear_index=linear_index, p=6)

    print('finish first p6 training')
    log_beta = torch.nn.Parameter(log_beta.clone().detach()).requires_grad_()

    optimizer2101 = torch.optim.LBFGS([log_beta],
                                      lr=0.1,
                                      max_iter=1000,
                                      max_eval=1000,
                                      line_search_fn='strong_wolfe')


    history8 = fit(num_epochs=1, optimizer=optimizer2101, cycle=2, verbose=False, beta=log_beta, beta_upper=beta_upper, beta_lower=beta_lower,
                   temp_K=temp_K, sig_save_stress=sig_save_stress, t_save_stress=t_save_stress,
                   sig_save_monotonic=sig_save_monotonic, eps_save_monotonic=eps_save_monotonic,
                   sig_save_cyclic=sig_save_cyclic, eps_save_cyclic=eps_save_cyclic,
                   strain_rate_save_monotonic=strain_rate_save_monotonic, strain_rate_save_stress=strain_rate_save_stress, strain_rate_save_cyclic=strain_rate_save_cyclic,
                   sig_mean_save=sig_mean_save, end_save_monotonic=end_save_monotonic, end_save_stress=end_save_stress, end_save_cyclic=end_save_cyclic, ranges=ranges, log_index=log_index, linear_index=linear_index, p=6)

    print('finish last p6 training')
    elapsed = tm.time() - start_time
    print('Training time: %.2f' % (elapsed))
    final_loss[idx_para] = history8[-1]
    # final_loss[idx_para] = history2[-1]
    training_time[idx_para] = elapsed

    # time_now = 'after'

    beta = log_beta.clone()
    beta[log_index] = betas_normalization.denormalize_log_betas(log_beta[log_index], ranges_log)
    beta[linear_index] = betas_normalization.denormalize_linear_betas(log_beta[linear_index], ranges_linear)
    par[idx_para, :] = beta
    '''
    np.savetxt(os.path.join(r'/cluster/home/haotxu/PINNs/save_inverse', time_now + '_' + str(index) + '_beta_range_para.txt'), par.cpu().detach().numpy(),delimiter=',')
    np.savetxt(os.path.join(r'/cluster/home/haotxu/PINNs/save_inverse', time_now + '_' + str(index) + '_beta_range_training_time.txt'), training_time.cpu().detach().numpy(), delimiter=',')
    np.savetxt(os.path.join(r'/cluster/home/haotxu/PINNs/save_inverse', time_now + '_' + str(index) + '_beta_range_loss.txt'), final_loss.cpu().detach().numpy(),delimiter=',')
    '''
np.savetxt(os.path.join(current_dir + r'\save_inverse', time_now + '_' + str(index) + '_beta_range_para.txt'), par.cpu().detach().numpy(),delimiter=',')
np.savetxt(os.path.join(current_dir + r'\save_inverse', time_now + '_' + str(index) + '_beta_range_training_time.txt'), training_time.cpu().detach().numpy(), delimiter=',')
np.savetxt(os.path.join(current_dir + r'\save_inverse', time_now + '_' + str(index) + '_beta_range_loss.txt'), final_loss.cpu().detach().numpy(),delimiter=',')

plt.show(block=True)
