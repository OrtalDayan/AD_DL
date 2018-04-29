
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='whitegrid')
#get_ipython().magic('matplotlib inline')
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')

# ## Plot all folds and splits learning curves

# In[ ]:


def plot_curves(results_folder):
    """Plot learning curves, accuracies and ROC AUC from epoch.

    Parameters
    ----------
    folder : string
        Location of the results .npy files.
    """
    for cvrand in range(5):
        for fold in range(5):
            eps = np.load('./{}/{}_{}nm_eps.npy'.format(results_folder,
                                                        cvrand, fold))
            tr_loss = np.load('./{}/{}_{}nm_tr_loss.npy'.format(results_folder,
                                                                cvrand, fold))
            vl_loss = np.load('./{}/{}_{}nm_vl_loss.npy'.format(results_folder,
                                                                cvrand, fold))
            vl_acc = np.load('./{}/{}_{}nm_vl_accs.npy'.format(results_folder,
                                                               cvrand, fold))
            vl_roc = np.load('./{}/{}_{}nm_vl_rocs.npy'.format(results_folder,
                                                               cvrand, fold))
            plt.figure(figsize=(15, 7))
            plt.plot(eps, tr_loss, label='train_loss')
            plt.plot(eps, vl_loss, label='val_loss')
            plt.plot(eps, vl_acc, label='val_accuracy')
            plt.plot(eps, vl_roc, label='val_rocauc')
            plt.ylim((0., 1.))
            plt.title('Nesterov')
            plt.legend(loc=0, frameon=True, framealpha=1)
            plt.show()


# In[ ]:


plot_curves('./results_resnet/ad_vs_norm')


# In[ ]:


plot_curves('./results_resnet/ad_vs_emci')


# In[ ]:


plot_curves('./results_resnet/ad_vs_lmci')


# In[ ]:


plot_curves('./results_resnet/emci_vs_norm')


# In[ ]:


plot_curves('./results_resnet/lmci_vs_norm')


# In[ ]:


plot_curves('./results_resnet/lmci_vs_emci')


# ## Plot ROC AUC curves with mean and std

# In[ ]:


def plot_auc(results_folder):
    """Plot ROC AUC curves with mean and std from epoch.

    Parameters
    ----------
    folder : string
        Location of the results .npy files.
    """
    eps = np.load('./{}/{}_{}nm_eps.npy'.format(results_folder,
                                                0, 0))
    vl_roc = np.zeros(eps.shape)
    for cvrand in range(5):
        for fold in range(5):
            eps = np.load('./{}/{}_{}nm_eps.npy'.format(results_folder,
                                                        cvrand, fold))
            vl_roc = np.vstack((vl_roc,
                                np.load('./{}/{}_{}nm_vl_rocs.npy'.format(
                                    results_folder, cvrand, fold))))
    plt.figure(figsize=(15, 7))
    plt.plot(eps, vl_roc[1:].mean(axis=0), label='val_rocauc')
    plt.fill_between(eps, vl_roc[1:].mean(axis=0) + vl_roc[1:].std(axis=0),
                     vl_roc[1:].mean(axis=0) - vl_roc[1:].std(axis=0),
                     alpha=.3)
    plt.ylim((0., 1.))
    plt.xlabel('Epoch')
    plt.legend(loc=0, frameon=True, framealpha=1)
    plt.show()
    print('AUC = {:.5f} +/- {:.5f}'.format(vl_roc[1:].mean(axis=0)[-1], vl_roc[1:].std(axis=0)[-1]))


# In[ ]:


plot_auc('./results_resnet/ad_vs_norm')


# In[ ]:


plot_auc('./results_resnet/ad_vs_emci')


# In[ ]:


plot_auc('./results_resnet/ad_vs_lmci')


# In[ ]:


plot_auc('./results_resnet/emci_vs_norm')


# In[ ]:


plot_auc('./results_resnet/lmci_vs_norm')


# In[ ]:


plot_auc('./results_resnet/lmci_vs_emci')

