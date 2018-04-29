
# coding: utf-8

# In[ ]:


import os

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# -----

# ## AD vs Normal

# In[ ]:


def plot(sm_folder, s):
    cv_results = []
    for i in range(5):
        # loading cv results
        with np.load(sm_folder + s + str(i) + '.npz') as f:
            cv_results.append([f['arr_%d' % i]
                               for i in range(len(f.files))][0])
    cv_results = np.array(cv_results)

    tmplate = 2  # accuracy
    plt.figure()
    plt.plot(cv_results[:, :, tmplate, :].mean(axis=0).mean(axis=0))
    y1 = cv_results[:, :, tmplate, :].mean(axis=0).mean(
        axis=0) + cv_results[:, :, tmplate, :].reshape(25, -1).std(axis=0)
    y2 = cv_results[:, :, tmplate, :].mean(axis=0).mean(
        axis=0) - cv_results[:, :, tmplate, :].reshape(25, -1).std(axis=0)
    plt.fill_between(np.arange(len(y1)), y1, y2, alpha=0.2)
    plt.title('mean accuracy ' + s[11:-1])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.savefig('mean_roc_auc__5_fold_'+ LABEL_1+'_vs_'+ LABEL_2 +'_'+str(i)+'.png')
    print('Accuracy: ' + str(cv_results[:, :, tmplate, :].mean(axis=0).mean(
        axis=0)[-1]) + ' ± ' + str(
        cv_results[:, :, tmplate, :].reshape(25, -1).std(axis=0)[-1]))

    tmplate = 3  # roc auc
    plt.figure()
    plt.plot(np.array(cv_results)[:, :, tmplate, :].mean(axis=0).mean(axis=0))
    y1 = np.array(cv_results)[:, :, tmplate, :].mean(axis=0).mean(
        axis=0) + np.array(cv_results)[:, :, tmplate, :].reshape(25, -1).std(
        axis=0)
    y2 = np.array(cv_results)[:, :, tmplate, :].mean(axis=0).mean(
        axis=0) - np.array(cv_results)[:, :, tmplate, :].reshape(25, -1).std(
        axis=0)
    plt.fill_between(np.arange(len(y1)), y1, y2, alpha=0.2)
    plt.title('mean roc_auc ' + s[11:-1])
    plt.ylabel('roc_auc')
    plt.xlabel('epoch')
    # plt.savefig('mean_roc_auc__5_fold_'+ LABEL_1+'_vs_'+ LABEL_2 +'_'+str(i)+'.png')
    print('ROC AUC: ' + str(
        np.array(cv_results)[:, :, tmplate, :].mean(axis=0).mean(
            axis=0)[-1]) + ' ± ' + str(
        np.array(cv_results)[:, :, tmplate, :].reshape(25, -1).std(axis=0)[-1]))


# _____

# ### AD vs Normal

# In[ ]:


sm_folder = './results/ad_vs_norm/'
s = 'cv_results_AD_vs_Normal_'
plot(sm_folder, s)


# _____

# ### AD LMCI

# In[ ]:


sm_folder = './results_cnn/ad_vs_lmci/'
s = 'cv_results_AD_vs_LMCI_'
plot(sm_folder, s)


# ______

# ### AD EMCI

# In[ ]:


s = 'cv_results_AD_vs_EMCI_'
sm_folder = './results_cnn/ad_vs_emci/'
plot(sm_folder, s)


# _______

# ### Normal vs EMCI

# In[ ]:


s = 'cv_results_Normal_vs_EMCI_'
sm_folder = './results_cnn/norm_vs_emci/'
plot(sm_folder, s)


# _______

# ### Normal vs LMCI

# In[ ]:


s = 'cv_results_Normal_vs_LMCI_'
sm_folder = './results_cnn/norm_vs_lmci/'
plot(sm_folder, s)


# ______

# ### EMCI vs LMCI

# In[ ]:


s = 'cv_results_EMCI_vs_LMCI_'
sm_folder = './results_cnn/emci_vs_lmci/'
plot(sm_folder, s)

