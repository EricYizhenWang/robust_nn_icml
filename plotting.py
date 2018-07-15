import numpy as np
import matplotlib.pyplot as plt

data_folder = './experiment_results/'
task = 'mnist'
flags = ['wb', 'wb_kernel', 'kernel', 'nn']
for flag in flags:
    fname = data_folder+task+flag+'new.npy'
    [standard, robust, at, at_all] = np.load(fname)
    print robust
    ep = [0.2*i for i in range(21)]
    fig, ax = plt.subplots()
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    axes = plt.gca()
    ymin = 0.1
    ymax = 1
    axes.set_ylim([ymin,ymax])
    l1 = ax.plot(ep, standard, marker = 's', label = 'StandardNN')
    l2 = ax.plot(ep, robust, marker = 'D', label = 'RobustNN')
    l3 = ax.plot(ep, at, marker = 'o', label = 'ATNN')
    l4 = ax.plot(ep, at_all, marker = 'o', label = 'ATNN-ALL')
    legend = ax.legend(loc = 'lower left', fontsize = 12)
    ax.set_ylabel('Classification Accuracy', fontsize = 18)
    ax.set_xlabel('Max $l_2$ Norm of Adv. Perturbation', fontsize = 18)
    if flag == 'wb' or flag=='kernel':
        ax.set_title('MNIST 1V7', fontsize = 20)
    fig.tight_layout()
    plt.savefig('./experiment_results/plots/'+task+'_'+flag+'.pdf')
    plt.show()
