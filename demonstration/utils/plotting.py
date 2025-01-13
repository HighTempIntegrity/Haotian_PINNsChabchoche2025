import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_train_loss(params, history):

    plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.plot(np.arange(1, len(history) + 1), history, label="Train Loss")
    plt.legend()
    plot_name = params[params['run']]["figure_name"] + '_' + params[
        'current_time'] + '_' + params['common_parameters'][
            'train_loss_figure_name']
    plot_path = os.path.join(params['common_parameters']['loss_save_path'],
                             plot_name)
    plt.savefig(plot_path)
    plt.close()


def plot_individual_losses(params, loss_11, loss_21, loss_31, loss_41):

    plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.plot(np.arange(1,
                       len(loss_11) + 1).reshape(-1, 1),
             np.log10(loss_11).reshape(-1, 1),
             label="Loss1")
    plt.plot(np.arange(1,
                       len(loss_21) + 1).reshape(-1, 1),
             np.log10(loss_21).reshape(-1, 1),
             label="Loss2")
    plt.plot(np.arange(1,
                       len(loss_31) + 1).reshape(-1, 1),
             np.log10(loss_31).reshape(-1, 1),
             label="Loss3")
    plt.plot(np.arange(1,
                       len(loss_41) + 1).reshape(-1, 1),
             np.log10(loss_41).reshape(-1, 1),
             label="Loss4")
    plt.legend()
    plot_name = params[params['run']]["figure_name"] + '_' + params[
        'current_time'] + '_' + params['common_parameters'][
            'pde_loss_figure_name']
    plot_path = os.path.join(params['common_parameters']['loss_save_path'],
                             plot_name)
    plt.savefig(plot_path)
    plt.close()


def plot_monotonic_learning_result(params, test_data):

    fig = plt.figure(figsize=(16, 8))

    eps_tot = test_data['eps_tot']
    sig_pred = test_data['sig_pred']
    sig_data = test_data['sig_data']
    eps_p_pred = test_data['eps_p_pred']
    eps_p_data = test_data['eps_p_data']
    eps_tot_data = test_data['eps_tot_data']
    R_pred = test_data['R_pred']
    R_data = test_data['R_data']
    X1_pred = test_data['X1_pred']
    X1_data = test_data['X1_data']
    X2_pred = test_data['X2_pred']
    X2_data = test_data['X2_data']

    ax1 = plt.subplot(1, 3, 1)
    ax1.set_ylabel('stress (MPa)')
    ax1.set_xlabel('log10(time) (s)')
    ax1.plot(eps_tot.cpu().detach().numpy(),
             sig_pred.cpu().detach().numpy(),
             ms=0.5,
             label="stress_predict")
    ax1.plot(eps_tot_data.cpu().detach().numpy(),
             sig_data.cpu().detach().numpy(),
             ms=0.5,
             label="stress_true")
    ax1.legend(loc='upper left')

    ax3 = plt.subplot(2, 3, 2)
    ax3.set_ylabel('eps_p')
    ax3.plot(eps_tot.cpu().detach().numpy(),
             eps_p_pred.cpu().detach().numpy(),
             ms=0.5,
             label="viscoplastic strain predict")
    ax3.plot(eps_tot_data.cpu().detach().numpy(),
             eps_p_data.cpu().detach().numpy(),
             ms=0.5,
             label="viscoplastic strain true")
    ax3.legend(loc='upper left')
    color = 'tab:red'
    ax12 = ax3.twinx()
    ax12.plot(eps_tot.cpu().detach().numpy(),
              test_data['loss1'].cpu().detach().numpy(),
              ms=0.5,
              label="Loss1",
              color=color)
    ax12.set_ylabel('loss1', color=color)
    ax12.legend(loc='upper right')
    ax12.tick_params(axis='y', labelcolor=color)
    plt.ylim((-10 * test_data['mean1'], 10 * test_data['mean1']))

    ax4 = plt.subplot(2, 3, 5, sharex=ax3)
    ax4.set_xlabel('log10(time) (s)')
    ax4.set_ylabel('R (MPa)')
    ax4.plot(eps_tot.cpu().detach().numpy(),
             R_pred.cpu().detach().numpy(),
             ms=0.5,
             label="R_predict")
    ax4.plot(eps_tot_data.cpu().detach().numpy(),
             R_data.cpu().detach().numpy(),
             ms=0.5,
             label="R_true")
    ax4.legend(loc='upper left')
    color = 'tab:red'
    ax42 = ax4.twinx()
    ax42.plot(eps_tot.cpu().reshape(-1, ).detach().numpy(),
              test_data['loss2'].cpu().reshape(-1, ).detach().numpy(),
              ms=0.5,
              label="Loss2",
              color=color)
    ax42.set_ylabel('loss2', color=color)
    ax42.legend(loc='upper right')
    ax42.tick_params(axis='y', labelcolor=color)
    plt.ylim((-10 * test_data['mean2'], 10 * test_data['mean2']))

    ax5 = plt.subplot(2, 3, 3)
    ax5.set_ylabel('X1 (MPa)')
    ax5.plot(eps_tot.cpu().detach().numpy(),
             X1_pred.cpu().detach().numpy(),
             ms=0.5,
             label="X1_predict")
    ax5.plot(eps_tot_data.cpu().detach().numpy(),
             X1_data.cpu().detach().numpy(),
             ms=0.5,
             label="X1_true")
    ax5.legend(loc='upper left')
    color = 'tab:red'
    ax52 = ax5.twinx()
    ax52.plot(eps_tot.cpu().detach().numpy(),
              test_data['loss3'].cpu().detach().numpy(),
              ms=0.5,
              label="Loss3",
              color=color)
    ax52.set_ylabel('loss3', color=color)
    ax52.legend(loc='upper right')
    ax52.tick_params(axis='y', labelcolor=color)
    plt.ylim((-10 * test_data['mean3'], 10 * test_data['mean3']))

    ax6 = plt.subplot(2, 3, 6, sharex=ax5)
    ax6.set_xlabel('log10(time) (s)')
    ax6.set_ylabel('X2 (MPa)')
    ax6.plot(eps_tot.cpu().detach().numpy(),
             X2_pred.cpu().detach().numpy(),
             ms=0.5,
             label="X2_predict")
    ax6.plot(eps_tot_data.cpu().detach().numpy(),
             X2_data.cpu().detach().numpy(),
             ms=0.5,
             label="X2_true")
    ax6.legend(loc='upper left')

    color = 'tab:red'
    ax62 = ax6.twinx()
    ax62.plot(eps_tot.cpu().detach().numpy(),
              test_data['loss4'].cpu().detach().numpy(),
              ms=0.5,
              label="Loss4",
              color=color)
    ax62.set_ylabel('loss4', color=color)
    ax62.legend(loc='upper right')
    ax62.tick_params(axis='y', labelcolor=color)
    plt.ylim((-10 * test_data['mean4'], 10 * test_data['mean4']))
    fig.tight_layout()

    plot_name = params[params['run']]["figure_name"] + '_' + params[
        'current_time'] + '_benchmark_' + str(
            test_data['index'] + 1) + '_rate_' + str(
                test_data['strain_rate']) + '.png'

    plot_full_path = os.path.join(
        params['common_parameters']['plot_save_path'], plot_name)

    plt.savefig(plot_full_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_stress_relaxation_learning_result(params, test_data):

    fig = plt.figure(figsize=(16, 8))

    t_plot = test_data['t_plot']
    t_data_plot = test_data['t_data_plot']
    sig_pred = test_data['sig_pred']
    sig_data = test_data['sig_data']
    t = test_data['t']
    t_data = test_data['t_data']
    R_pred = test_data['R_pred']
    R_data = test_data['R_data']
    X1_pred = test_data['X1_pred']
    X1_data = test_data['X1_data']
    X2_pred = test_data['X2_pred']
    X2_data = test_data['X2_data']

    ax1 = plt.subplot(2, 3, 1)
    ax1.set_ylabel('stress (MPa)')
    ax1.set_xlabel('log10(time) (s)')
    ax1.plot(t_plot.cpu().detach().numpy(),
             sig_pred.cpu().detach().numpy(),
             ms=0.5,
             label="stress_predict")
    ax1.plot(t_data_plot.cpu().detach().numpy(),
             sig_data.cpu().detach().numpy(),
             ms=0.5,
             label="stress_true")
    ax1.legend(loc='upper left')

    ax2 = plt.subplot(2, 3, 4)
    ax2.set_ylabel('stress (MPa)')
    ax2.set_xlabel('time (s)')
    ax2.plot(t.cpu().detach().numpy(),
             sig_pred.cpu().detach().numpy(),
             ms=0.5,
             label="stress_predict")
    ax2.plot(t_data.cpu().detach().numpy(),
             sig_data.cpu().detach().numpy(),
             ms=0.5,
             label="stress_true")
    ax2.legend(loc='upper left')

    ax3 = plt.subplot(2, 3, 2)
    ax3.set_ylabel('eps_p')
    ax3.plot(t_plot.cpu().detach().numpy(),
             test_data['eps_p_pred'].cpu().detach().numpy(),
             ms=0.5,
             label="viscoplastic strain predict")
    ax3.plot(t_data_plot.cpu().detach().numpy(),
             test_data['eps_p_data'].cpu().detach().numpy(),
             ms=0.5,
             label="viscoplastic strain true")
    ax3.legend(loc='upper left')
    color = 'tab:red'
    ax12 = ax3.twinx()
    ax12.plot(t_plot.cpu().detach().numpy(),
              test_data['loss1'].cpu().detach().numpy(),
              ms=0.5,
              label="Loss1",
              color=color)
    ax12.set_ylabel('loss1', color=color)
    ax12.legend(loc='upper right')
    ax12.tick_params(axis='y', labelcolor=color)
    plt.ylim((-10 * test_data['mean1'], 10 * test_data['mean1']))

    ax4 = plt.subplot(2, 3, 5, sharex=ax3)
    ax4.set_xlabel('log10(time) (s)')
    ax4.set_ylabel('R (MPa)')
    ax4.plot(t_plot.cpu().detach().numpy(),
             R_pred.cpu().detach().numpy(),
             ms=0.5,
             label="R_predict")
    ax4.plot(t_data_plot.cpu().detach().numpy(),
             R_data.cpu().detach().numpy(),
             ms=0.5,
             label="R_true")
    ax4.legend(loc='upper left')
    color = 'tab:red'
    ax42 = ax4.twinx()
    ax42.plot(t_plot.cpu().reshape(-1, ).detach().numpy(),
              test_data['loss2'].cpu().reshape(-1, ).detach().numpy(),
              ms=0.5,
              label="Loss2",
              color=color)
    ax42.set_ylabel('loss2', color=color)
    ax42.legend(loc='upper right')
    ax42.tick_params(axis='y', labelcolor=color)
    plt.ylim((-10 * test_data['mean2'], 10 * test_data['mean2']))

    ax5 = plt.subplot(2, 3, 3)
    ax5.set_ylabel('X1 (MPa)')
    ax5.plot(t_plot.cpu().detach().numpy(),
             X1_pred.cpu().detach().numpy(),
             ms=0.5,
             label="X1_predict")
    ax5.plot(t_data_plot.cpu().detach().numpy(),
             X1_data.cpu().detach().numpy(),
             ms=0.5,
             label="X1_true")
    ax5.legend(loc='upper left')
    color = 'tab:red'
    ax52 = ax5.twinx()
    ax52.plot(t_plot.cpu().detach().numpy(),
              test_data['loss3'].cpu().detach().numpy(),
              ms=0.5,
              label="Loss3",
              color=color)
    ax52.set_ylabel('loss3', color=color)
    ax52.legend(loc='upper right')
    ax52.tick_params(axis='y', labelcolor=color)
    plt.ylim((-10 * test_data['mean3'], 10 * test_data['mean3']))

    ax6 = plt.subplot(2, 3, 6, sharex=ax5)
    ax6.set_xlabel('log10(time) (s)')
    ax6.set_ylabel('X2 (MPa)')
    ax6.plot(t_plot.cpu().detach().numpy(),
             X2_pred.cpu().detach().numpy(),
             ms=0.5,
             label="X2_predict")
    ax6.plot(t_data_plot.cpu().detach().numpy(),
             X2_data.cpu().detach().numpy(),
             ms=0.5,
             label="X2_true")
    ax6.legend(loc='upper left')

    color = 'tab:red'
    ax62 = ax6.twinx()
    ax62.plot(t_plot.cpu().detach().numpy(),
              test_data['loss4'].cpu().detach().numpy(),
              ms=0.5,
              label="Loss4",
              color=color)
    ax62.set_ylabel('loss4', color=color)
    ax62.legend(loc='upper right')
    ax62.tick_params(axis='y', labelcolor=color)
    plt.ylim((-10 * test_data['mean4'], 10 * test_data['mean4']))
    fig.tight_layout()

    plot_name = params[params['run']]["figure_name"] + '_' + params[
        'current_time'] + '_benchmark_' + str(
            test_data['index'] + 1) + '_rate_' + str(
                test_data['strain_rate']) + '.png'

    plot_full_path = os.path.join(
        params['common_parameters']['plot_save_path'], plot_name)

    plt.savefig(plot_full_path, dpi=300, bbox_inches='tight')
    plt.close()
