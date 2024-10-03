import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot(x, x_name, y_left, y_left_name, y_right, y_right_name):
    fig, ax1 = plt.subplots(figsize=(10,7))
    ax1.plot(x, y_left)
    ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
    ax1.set_ylabel(y_left_name, color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.set_ylim(0.5, 0.8)

    ax2 = ax1.twinx()
    ax2.plot(x, y_right, color='r')
    ax2.set_ylabel(y_right_name, color='r', fontsize=16, fontweight='bold')
    if 'DI' in y_right_name:
        ax2.set_ylim(0., 0.7)
    else:
        ax2.set_ylim(-0.25, 0.1)

    best_ind = np.argmax(y_left)
    ax2.axvline(np.array(x)[best_ind], color='k', linestyle=':')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)


def plot_algo_lr(orig_metrics_mean, transf_metrics_mean, dir_metrics_mean, reweigh_metrics_mean, egr_metrics_mean, pr_orig_metrics_mean, cpp_metrics_mean, ro_metrics_mean,
              orig_error_metrics, transf_error_metrics, dir_error_metrics, reweigh_error_metrics, egr_error_metrics, pr_orig_error_metrics, cpp_error_metrics, ro_error_metrics,
              model_type):
    pd.set_option('display.multi_sparse', False)
    plt.rcParams.update({'font.size': 8}) # must set in top

    results = [orig_metrics_mean,
            transf_metrics_mean,
            dir_metrics_mean,
            reweigh_metrics_mean,
            egr_metrics_mean,
            pr_orig_metrics_mean,
            cpp_metrics_mean,
            ro_metrics_mean]


    errors = [orig_error_metrics,
            transf_error_metrics,
            dir_error_metrics,
            reweigh_error_metrics,
            egr_error_metrics,
            pr_orig_error_metrics,
            cpp_error_metrics,
            ro_error_metrics]

    index = pd.Series([model_type+'_orig']+ [model_type+'_syn']+ [model_type+'_dir']+ [model_type+'_rew']+ [model_type+'_egr']+ [model_type+'_pr']+ [model_type+'_cpp']+ [model_type+'_ro'], name='Classifier Bias Mitigator')

    df = pd.concat([pd.DataFrame(metrics) for metrics in results], axis=0).set_index(index)
    df_error = pd.concat([pd.DataFrame(metrics) for metrics in errors], axis=0).set_index(index)
    ax = df.plot.bar(yerr=df_error, capsize=4, rot=0, subplots=True, title=['','','','','', '', '', '', '', ''], fontsize = 12)
    plot1 = ax[0]
    plot1.set_ylim=([0, 0.8])
    plot2 = ax[1]
    plot2.set_ylim=([-0.5, 0])
    plot3 = ax[2]
    plot3.set_ylim=([0, 1])
    plot4 = ax[3]
    plot4.set_ylim=([-0.5, 0])
    plot5 = ax[4]
    plot5.set_ylim=([-0.5, 0])
    plot5 = ax[5]
    plot5.set_ylim=([0, 0.2])

    plt.legend(bbox_to_anchor=(1.5, 1.0))
    # plt.savefig('exp_dir_rew_egr_pr_cpp_ro.jpg')
    print(df)

def plot_algo(orig_metrics_mean, transf_metrics_mean, dir_metrics_mean, reweigh_metrics_mean, egr_metrics_mean, cpp_metrics_mean, ro_metrics_mean,
              orig_error_metrics, transf_error_metrics, dir_error_metrics, reweigh_error_metrics, egr_error_metrics, cpp_error_metrics, ro_error_metrics,
              model_type):
    pd.set_option('display.multi_sparse', False)
    plt.rcParams.update({'font.size': 8}) # must set in top

    results = [orig_metrics_mean,
            transf_metrics_mean,
            dir_metrics_mean,
            reweigh_metrics_mean,
            egr_metrics_mean,
            cpp_metrics_mean,
            ro_metrics_mean]


    errors = [orig_error_metrics,
            transf_error_metrics,
            dir_error_metrics,
            reweigh_error_metrics,
            egr_error_metrics,
            cpp_error_metrics,
            ro_error_metrics]

    index = pd.Series([model_type+'_orig']+ [model_type+'_syn']+ [model_type+'_dir']+ [model_type+'_rew']+ [model_type+'_egr']+ [model_type+'_cpp'], name='Classifier Bias Mitigator')

    df = pd.concat([pd.DataFrame(metrics) for metrics in results], axis=0).set_index(index)
    df_error = pd.concat([pd.DataFrame(metrics) for metrics in errors], axis=0).set_index(index)
    ax = df.plot.bar(yerr=df_error, capsize=4, rot=0, subplots=True, title=['','','','','', '', '', '', '', ''], fontsize = 12)
    plot1 = ax[0]
    plot1.set_ylim=([0, 0.8])
    plot2 = ax[1]
    plot2.set_ylim=([-0.5, 0])
    plot3 = ax[2]
    plot3.set_ylim=([0, 1])
    plot4 = ax[3]
    plot4.set_ylim=([-0.5, 0])
    plot5 = ax[4]
    plot5.set_ylim=([-0.5, 0])
    plot5 = ax[5]
    plot5.set_ylim=([0, 0.2])

    #plt.legend(bbox_to_anchor=(1.5, 1.0))
    #plt.savefig('./eps/'+model_type+'_dir_rew_egr_cpp_ro.eps', format='eps')
    print(df)

