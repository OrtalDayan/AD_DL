
# coding: utf-8

# ## Notebook for reproduce paper results
# 
# ##### content
# 
# - [Batch iteration functions](#Batch-iteration-functions)
# - [Train functions](#Train-functions)
# - [Network architecture](#network-architecture)
# - [Cross_validation function](#cross-validation)
# - [Cross-validation one_vs_one - run](#Cross-validation-one_vs_one)
# 
# 
# 
# 

# In[1]:


import os
import gc
import sys
import time
import datetime
import traceback
from collections import OrderedDict

import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import *

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers.dnn import Conv3DDNNLayer
from lasagne.layers.dnn import Pool3DDNNLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax, identity
from lasagne.layers import set_all_param_values
from lasagne.layers import DropoutLayer

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import _pickle

# Turn interactive plotting off - I added
plt.ioff()

# In[ ]:


PATH_TO_REP = 'data/'  # adni_data

inp_shape = (None, 1, 110, 110, 110)


# In[2]:


input_var = T.TensorType('float32', (False,) * 5)('inputs')
target_var = T.ivector('targets')


# ____

# ### Batch iteration functions
# 

# In[3]:


from utils import iterate_minibatches, iterate_minibatches_train


# ### Train functions

# In[4]:


def get_train_functions(nn, updates_method=lasagne.updates.nesterov_momentum,
                        _lr=0.00001):
    """
    Return functions for training, validation network and predicting answers.

    Parameters
    ----------
    nn : lasagne.Layer
        network last layer

    updates_method : function
        like in lasagne.objectives or function from there

    _lr : float
        learning rate which relate with the updates_method

    Returns
    -------
    train_fn : theano.function
        Train network function.
    val_fn : theano.function
        Validation function.
    pred_fn : theano.function
        Function for get predicts from network.
    """
    prediction = lasagne.layers.get_output(nn)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(nn, trainable=True)
    updates = updates_method(loss, params, learning_rate=_lr)

    test_prediction = lasagne.layers.get_output(nn, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], test_loss)
    pred_fn = theano.function([input_var], test_prediction)

    # I added
    # save the functions to disk
    #with open(train_fn, 'wb') as pickle_file:
    #    _pickle.dump(train_fn, pickle_file)

    #filename = 'train_fn.sav'
    #pickle.dump(train_fn, filename)

    #filename = 'val_fn.sav'
    #pickle.dump(val_fn, filename)

    #filename = 'pred_fn.sav'
    #pickle.dump(pred_fn, filename)
    # up to hear

    return train_fn, val_fn, pred_fn


# In[5]:


def train(train_fn, val_fn, test_fn,
          X_train, y_train,
          X_test, y_test,
          LABEL_1, LABEL_2,  # labels of the y.
          num_epochs=100, batchsize=5,
          dict_of_paths={'output': '1.txt', 'picture': '1.png',
                         'report': 'report.txt'},
          report='''trained next architecture, used some
                    optimizstion method with learning rate...''',
          architecture='nn=...'):
    """
    Iterate minibatches on train subset and validate results on test subset.

    Parameters
    ----------
    train_fn : theano.function
        Train network function.
    val_fn : theano.function
        Validation network function.
    test_fn : theano.function
        Function for get predicts from network.
    X_train : numpy array
        X train subset.
    y_train : numpy array
        Y train subset.
    X_test : numpy array
        X test subset.
    y_test : numpy array
        Y test subset.
    LABEL_1 : {'AD', 'LMCI', 'EMCI', 'Normal'}
        String label for target == 0.
    LABEL_2 : {'AD', 'LMCI', 'EMCI', 'Normal'}
        String label for target == 1.
    dict_of_paths : dictionary
        Names of files to store results.
    report : string
        Some comments which will saved into report after ending of training.
    num_epochs : integer
        Number of epochs for all of the experiments. Default is 100.
    batchsize : integer
        Batchsize for network training. Default is 5.

    Returns
    -------
    tr_losses : numpy.array
        Array with loss values on train.
    val_losses : numpy.array
        Array with loss values on test.
    val_accs : numpy.array
        Array with accuracy values on test.
    rocs : numpy.array
        Array with roc auc values on test.

    """

    eps = []
    tr_losses = []
    val_losses = []
    val_accs = []
    rocs = []

    FILE_PATH = dict_of_paths['output']
    PICTURE_PATH = dict_of_paths['picture']
    REPORT_PATH = dict_of_paths['report']

    # here we written outputs on each step (val and train losses, accuracy, auc)
    with open(FILE_PATH, 'w') as f:
        f.write('\n----------\n\n' + str(datetime.datetime.now())[:19])
        f.write('\n' + LABEL_1 + '-' + LABEL_2 + '\n')
        f.close()

    # starting training
    print("Starting training...", flush=True)
    den = X_train.shape[0] / batchsize
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches_train(X_train, y_train, batchsize,
                                               shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_batches = 0
        preds = []
        targ = []
        for batch in iterate_minibatches(X_test, y_test, batchsize,
                                         shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1
            out = test_fn(inputs)
            [preds.append(i) for i in out]
            [targ.append(i) for i in targets]

        preds_tst = np.array(preds).argmax(axis=1)
        ##
        ## output
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1,
                                                   num_epochs,
                                                   time.time() - start_time),
              flush=True)
        print("  training loss:\t\t{:.7f}".format(train_err / train_batches),
              flush=True)
        print("  validation loss:\t\t{:.7f}".format(val_err / val_batches),
              flush=True)
        print('  validation accuracy:\t\t{:.7f}'.format(
            accuracy_score(np.array(targ),
                           preds_tst)), flush=True)
        print('Confusion matrix for test:', flush=True)
        print(confusion_matrix(np.array(targ), np.array(preds).argmax(axis=1)),
              flush=True)
        rcs = roc_auc_score(np.array(targ), np.array(preds)[:, 1])
        sys.stderr.write('Pairwise ROC_AUCs: ' + str(rcs))
        print('')

        with open(FILE_PATH, 'a') as f:
            f.write("\nEpoch {} of {} took {:.3f}s".format(epoch + 1,
                                                           num_epochs,
                                                           time.time() - start_time))
            f.write(
                "\n training loss:\t\t{:.7f}".format(train_err / train_batches))
            f.write(
                "\n validation loss:\t\t{:.7f}".format(val_err / val_batches))
            f.write('\n validation accuracy:\t\t{:.7f}'.format(
                accuracy_score(np.array(targ),
                               np.array(preds).argmax(axis=1))))

            f.write('\n Pairwise ROC_AUCs:' + str(rcs) + '\n')
            f.close()
        ## output
        ## saving results
        eps.append(epoch + 1)
        tr_losses.append(train_err / train_batches)
        val_losses.append(val_err / val_batches)
        val_accs.append(
            accuracy_score(np.array(targ), np.array(preds).argmax(axis=1)))
        rocs.append(rcs)

    print('ended!')

    ### and save plots
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.title('Loss ' + LABEL_1 + ' vs ' + LABEL_2)
    plt.xlabel('Epoch')
    plt.ylim((0, 3))
    plt.ylabel('Loss')
    plt.plot(eps, tr_losses, label='train')
    plt.plot(eps, val_losses, label='validation')
    plt.legend(loc=0)
    #
    plt.subplot(2, 2, 2)
    plt.title('Accuracy ' + LABEL_1 + ' vs ' + LABEL_2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(eps, val_accs, label='validation accuracy')
    plt.legend(loc=0)
    #
    plt.subplot(2, 2, 3)
    plt.title('AUC ' + LABEL_1 + ' vs ' + LABEL_2)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.plot(eps, np.array(rocs), label='validation auc')
    plt.legend(loc=0)
    #
    plt.subplot(2, 2, 4)
    plt.title('architecture')
    plt.axis('off')
    plt.text(0, -0.1, architecture, fontsize=7, )
    plt.savefig(PICTURE_PATH)
    ###########

    # write that trainig was ended
    with open(FILE_PATH, 'a') as f:
        f.write('\nended at ' + str(datetime.datetime.now())[:19] + '\n \n')
        f.close()

    # write report
    with open(REPORT_PATH, 'a') as f:
        f.write(
            '\n классификация ' + LABEL_1 + ' vs ' + LABEL_2 + '\n' + report)
        #         f.write(architecture)
        f.write('final results are:')
        f.write('\n tr_loss: ' + str(tr_losses[-1]) + '\n val_loss: ' +                 str(val_losses[-1]) + '\n val_acc; ' + str(val_accs[-1]) +                 '\n val_roc_auc: ' + str(rocs[-1]))
        f.write('\nresults has been saved in files:\n')
        f.write(FILE_PATH + '\n')
        f.write(PICTURE_PATH + '\n')
        f.write('\n ___________________ \n\n\n')
        f.close()

    return tr_losses, val_losses, val_accs, rocs


# ______

# ### network architecture

# In[6]:


def build_net():
    """Method for VGG like net Building.

    Returns
    -------
    nn : lasagne.layer
        Network.
    """
    nn = InputLayer(inp_shape, input_var=input_var)

    nn = Conv3DDNNLayer(nn, 8, 3)
    nn = Conv3DDNNLayer(nn, 8, 3, nonlinearity=identity)
    nn = NonlinearityLayer(nn)
    nn = Pool3DDNNLayer(nn, 2)

    nn = Conv3DDNNLayer(nn, 16, 3)
    nn = Conv3DDNNLayer(nn, 16, 3, nonlinearity=identity)
    nn = NonlinearityLayer(nn)
    nn = Pool3DDNNLayer(nn, 2)

    nn = Conv3DDNNLayer(nn, 32, 3)
    nn = Conv3DDNNLayer(nn, 32, 3)
    nn = Conv3DDNNLayer(nn, 32, 3, nonlinearity=identity)
    nn = NonlinearityLayer(nn)
    nn = Pool3DDNNLayer(nn, 2)

    nn = Conv3DDNNLayer(nn, 64, 3)
    nn = Conv3DDNNLayer(nn, 64, 3)
    nn = Conv3DDNNLayer(nn, 64, 3, nonlinearity=identity)
    nn = NonlinearityLayer(nn)
    nn = Pool3DDNNLayer(nn, 2)

    nn = DenseLayer(nn, num_units=128)
    nn = BatchNormLayer(nn)
    nn = DropoutLayer(nn, p=0.7)

    nn = DenseLayer(nn, num_units=64)

    nn = DenseLayer(nn, num_units=2,
                    nonlinearity=lasagne.nonlinearities.softmax)

    # save the model to disk - I added
    filename = 'VoxCNN.sav'
    joblib.dump(nn, filename)

    return nn


# writing architecture in report
architecture = '''
nn = InputLayer(inp_shape, input_var=input_var)

nn = Conv3DDNNLayer(nn, 8, 3)
nn = Conv3DDNNLayer(nn, 8, 3, nonlinearity=identity)
nn = NonlinearityLayer(nn)
nn = Pool3DDNNLayer(nn, 2)

nn = Conv3DDNNLayer(nn, 16, 3)
nn = Conv3DDNNLayer(nn, 16, 3, nonlinearity=identity)
nn = NonlinearityLayer(nn)
nn = Pool3DDNNLayer(nn, 2)

nn = Conv3DDNNLayer(nn, 32, 3)
nn = Conv3DDNNLayer(nn, 32, 3)
nn = Conv3DDNNLayer(nn, 32, 3, nonlinearity=identity)
nn = NonlinearityLayer(nn)
nn = Pool3DDNNLayer(nn, 2)

nn = Conv3DDNNLayer(nn, 64, 3)
nn = Conv3DDNNLayer(nn, 64, 3)
nn = Conv3DDNNLayer(nn, 64, 3, nonlinearity=identity)
nn = NonlinearityLayer(nn)
nn = Pool3DDNNLayer(nn, 2)

nn = DenseLayer(nn, num_units=128)
nn = BatchNormLayer(nn)
nn = DropoutLayer(nn, p=0.7)

nn = DenseLayer(nn, num_units=64)

nn = DenseLayer(nn, num_units=2, nonlinearity=lasagne.nonlinearities.softmax)

'''


# _____

# ### cross validation

# In[7]:


rnd_states = [14, 11, 1993, 19931411, 14111993]


# In[8]:


def run_cross_validation(LABEL_1, LABEL_2, results_folder):
    """
    Method for cross-validation.
    Takes two labels, reading data, prepair data with this labels for trainig.

    Parameters
    ----------
    LABEL_1 : {'AD', 'LMCI', 'EMCI', 'Normal'}
        String label for target == 0.
    LABEL_2 : {'AD', 'LMCI', 'EMCI', 'Normal'}
        String label for target == 1.
    results_folder : string
        Folder to store results.

    Returns
    -------
    None.
    """
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # reading data
    gc.collect()
    metadata = pd.read_csv(PATH_TO_REP + 'metadata.csv')
    metadata.columns = ['Index','Label', 'Path'] # I added to fix error 
    smc_mask = (
    (metadata.Label == LABEL_1) | (metadata.Label == LABEL_2)).values.astype(
        'bool')
    y = (metadata[smc_mask].Label == LABEL_1).astype(np.int32).values
    data = np.zeros((smc_mask.sum(), 1, 110, 110, 110), dtype='float32')
    # into memory
    for it, im in tqdm(enumerate(metadata[smc_mask].Path.values),
                       total=smc_mask.sum(), desc='Reading MRI to memory'):
        mx = nib.load(im).get_data().max(axis=0).max(axis=0).max(axis=0)
        data[it, 0, :, :, :] = np.array(nib.load(im).get_data()) / mx

    # loop by random states (different splitting)
    for i in range(len(rnd_states)):
        counter = 1
        cv_results = []
        skf = StratifiedKFold(y, n_folds=5, random_state=rnd_states[i])

        for tr, ts in skf:
            X_train = data[tr]
            X_test = data[ts]
            y_train = y[tr]
            y_test = y[ts]
            # creating folder for random states
            rnd_state_folder = results_folder + 'rnd_state_' + str(i) + '/'
            if not os.path.exists(rnd_state_folder):
                os.makedirs(rnd_state_folder)

            dict_of_paths = {
                'output': rnd_state_folder + 'Exp_CV_' + str(
                    counter) + '_' + LABEL_1 + '_vs_' + \
                          LABEL_2 + '_.txt',
                'picture': rnd_state_folder + 'Exp_CV_' + str(
                    counter) + '_' + LABEL_1 + '_vs_' + \
                           LABEL_2 + '_.png',
                'report': 'report.txt'
            }

            report = '\n' + LABEL_1 + '_vs_' + LABEL_2 + 'cv_fold ' +                      str(counter) + ' random state ' + str(i) +                      '_\n' + 'adam, lr=0.000027' + '\n '
            # building net and training
            nn = build_net()
            train_fn, val_fn, test_fn = get_train_functions(nn,
                                                            updates_method=lasagne.updates.adam,
                                                            _lr=0.000027)

            try:
                tr_losses, val_losses, val_accs, rocs = train(train_fn, val_fn,
                                                              test_fn, X_train,
                                                              y_train, X_test,
                                                              y_test, LABEL_1,
                                                              LABEL_2,
                                                              num_epochs=150,
                                                              batchsize=5,
                                                              dict_of_paths=dict_of_paths,
                                                              report=report,
                                                              architecture=architecture)
                cv_results.append((tr_losses, val_losses, val_accs, rocs))
            except Exception as e:
                with open('errors_msg.txt', 'a') as f:
                    f.write('Time: ' + str(datetime.datetime.now())[:19] +                             '\n' + str(e) + traceback.format_exc())

            counter += 1
            # saving network params
        #             np.savez('net_weights'+ str(counter) + str(i) +'.npz',
        #                      *lasagne.layers.get_all_param_values(nn))

        # saving losses, aucs, accuracies
        np.savez(results_folder + 'cv_results_' + LABEL_1 +                  '_vs_' + LABEL_2 + '_' + str(i) + '.npz', np.array(cv_results))

        # plotting mean roc_auc and  with losses by random_state
        plt.figure()
        plt.plot(np.array(cv_results)[:, 3, :].mean(axis=0))
        y1 = np.array(cv_results)[:, 3, :].mean(axis=0) + np.array(cv_results)[
                                                          :, 3, :].std(axis=0)
        y2 = np.array(cv_results)[:, 3, :].mean(axis=0) - np.array(cv_results)[
                                                          :, 3, :].std(axis=0)
        plt.fill_between(np.arange(len(y1)), y1, y2, alpha=0.4)
        plt.title(
            'mean roc auc' + '_' + str(i) + '_ ' + LABEL_1 + ' vs ' + LABEL_2)
        plt.ylabel('roc_auc')
        plt.xlabel('epoch')
        plt.savefig(
            rnd_state_folder + 'mean_roc_auc_5_fold_cv_' + LABEL_1 + '_vs_' + LABEL_2 + \
            '_for_rnd_state_' + str(i) + '.png')


# -------

# ### Cross-validation one_vs_one
# In[9]:


run_cross_validation('AD', 'Normal', './results_cnn/ad_vs_norm/')


# In[10]:


run_cross_validation('AD', 'LMCI', './results_cnn/ad_vs_lmci/')


# In[11]:


run_cross_validation('AD', 'EMCI', './results_cnn/ad_vs_emci/')


# In[12]:


run_cross_validation('Normal', 'EMCI', './results_cnn/norm_vs_emci/')


# In[13]:


run_cross_validation('Normal', 'LMCI', './results_cnn/norm_vs_lmci/')


# In[14]:


run_cross_validation('EMCI', 'LMCI', './results_cnn/emci_vs_lmci/')

