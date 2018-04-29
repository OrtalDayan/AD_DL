import numpy as np
import pandas as pd

vt = ['vl_accs', 'vl_rocs']
results_directory = ['ad_vs_norm','ad_vs_emci','ad_vs_lmci','emci_vs_norm','lmci_vs_norm','lmci_vs_emci']

j=0
for rd in results_directory:
    print rd + ':'
    val = np.load('./results_resnet/{}/{}_{}nm_eps.npy'.format(results_directory[0], 0, 0))
    val_accs = np.zeros(val.shape)
    val_roc = np.zeros(val.shape)

    for cvrand in range(5):

        for fold in range(5):
            val_accs = np.vstack((val_accs, np.load('./results_resnet/{}/{}_{}nm_{}.npy'.format(results_directory[j], cvrand, fold, vt[0]))))
            val_roc = np.vstack((val_roc, np.load('./results_resnet/{}/{}_{}nm_{}.npy'.format(results_directory[j], cvrand, fold, vt[1]))))
    j = j+1

    value_ac = val_accs[1:].mean(axis=0)[-1]
    std_ac = val_accs[1:].std(axis=0)[-1]
    print('{} = {:.2f} +/- {:.2f}'.format(vt[0], value_ac, std_ac))

    value_roc = val_roc[1:].mean(axis=0)[-1]
    std_roc = val_roc[1:].std(axis=0)[-1]
    print('{} = {:.2f} +/- {:.2f}'.format(vt[1], value_roc, std_roc))



