import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm
import logging
from sklearn.cross_validation import StratifiedKFold
import lasagne
import theano
from lasagne.layers import InputLayer
from lasagne.layers.dnn import Conv3DDNNLayer
from lasagne.layers.dnn import Pool3DDNNLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import identity, softmax
from lasagne.objectives import categorical_crossentropy
import theano.tensor as T
import time
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import sys
import os

from utils import iterate_minibatches, iterate_minibatches_train

input_var = T.tensor5(name='input', dtype='float32')
target_var = T.ivector()

def build_net():
    """Method for VoxResNet Building.

    Returns
    -------
    dictionary
        Network dictionary.
    """
    net = {}
    net['input'] = InputLayer((None, 1, 110, 110, 110), input_var=input_var)
    net['conv1a'] = Conv3DDNNLayer(net['input'], 32, 3, pad='same',
                                   nonlinearity=identity)
    net['bn1a'] = BatchNormLayer(net['conv1a'])
    net['relu1a'] = NonlinearityLayer(net['bn1a'])
    net['conv1b'] = Conv3DDNNLayer(net['relu1a'], 32, 3, pad='same',
                                   nonlinearity=identity)
    net['bn1b'] = BatchNormLayer(net['conv1b'])
    net['relu1b'] = NonlinearityLayer(net['bn1b'])
    net['conv1c'] = Conv3DDNNLayer(net['relu1b'], 64, 3, stride=(2, 2, 2),
                                   pad='same', nonlinearity=identity)
    # VoxRes block 2
    net['voxres2_bn1'] = BatchNormLayer(net['conv1c'])
    net['voxres2_relu1'] = NonlinearityLayer(net['voxres2_bn1'])
    net['voxres2_conv1'] = Conv3DDNNLayer(net['voxres2_relu1'], 64, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres2_bn2'] = BatchNormLayer(net['voxres2_conv1'])
    net['voxres2_relu2'] = NonlinearityLayer(net['voxres2_bn2'])
    net['voxres2_conv2'] = Conv3DDNNLayer(net['voxres2_relu2'], 64, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres2_out'] = ElemwiseSumLayer([net['conv1c'],
                                           net['voxres2_conv2']])
    # VoxRes block 3
    net['voxres3_bn1'] = BatchNormLayer(net['voxres2_out'])
    net['voxres3_relu1'] = NonlinearityLayer(net['voxres3_bn1'])
    net['voxres3_conv1'] = Conv3DDNNLayer(net['voxres3_relu1'], 64, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres3_bn2'] = BatchNormLayer(net['voxres3_conv1'])
    net['voxres3_relu2'] = NonlinearityLayer(net['voxres3_bn2'])
    net['voxres3_conv2'] = Conv3DDNNLayer(net['voxres3_relu2'], 64, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres3_out'] = ElemwiseSumLayer([net['voxres2_out'],
                                           net['voxres3_conv2']])

    net['bn4'] = BatchNormLayer(net['voxres3_out'])
    net['relu4'] = NonlinearityLayer(net['bn4'])
    net['conv4'] = Conv3DDNNLayer(net['relu4'], 64, 3, stride=(2, 2, 2),
                                  pad='same', nonlinearity=identity)
    # VoxRes block 5
    net['voxres5_bn1'] = BatchNormLayer(net['conv4'])
    net['voxres5_relu1'] = NonlinearityLayer(net['voxres5_bn1'])
    net['voxres5_conv1'] = Conv3DDNNLayer(net['voxres5_relu1'], 64, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres5_bn2'] = BatchNormLayer(net['voxres5_conv1'])
    net['voxres5_relu2'] = NonlinearityLayer(net['voxres5_bn2'])
    net['voxres5_conv2'] = Conv3DDNNLayer(net['voxres5_relu2'], 64, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres5_out'] = ElemwiseSumLayer([net['conv4'], net['voxres5_conv2']])
    # VoxRes block 6
    net['voxres6_bn1'] = BatchNormLayer(net['voxres5_out'])
    net['voxres6_relu1'] = NonlinearityLayer(net['voxres6_bn1'])
    net['voxres6_conv1'] = Conv3DDNNLayer(net['voxres6_relu1'], 64, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres6_bn2'] = BatchNormLayer(net['voxres6_conv1'])
    net['voxres6_relu2'] = NonlinearityLayer(net['voxres6_bn2'])
    net['voxres6_conv2'] = Conv3DDNNLayer(net['voxres6_relu2'], 64, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres6_out'] = ElemwiseSumLayer([net['voxres5_out'],
                                           net['voxres6_conv2']])

    net['bn7'] = BatchNormLayer(net['voxres6_out'])
    net['relu7'] = NonlinearityLayer(net['bn7'])
    net['conv7'] = Conv3DDNNLayer(net['relu7'], 128, 3, stride=(2, 2, 2),
                                  pad='same', nonlinearity=identity)

    # VoxRes block 8
    net['voxres8_bn1'] = BatchNormLayer(net['conv7'])
    net['voxres8_relu1'] = NonlinearityLayer(net['voxres8_bn1'])
    net['voxres8_conv1'] = Conv3DDNNLayer(net['voxres8_relu1'], 128, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres8_bn2'] = BatchNormLayer(net['voxres8_conv1'])
    net['voxres8_relu2'] = NonlinearityLayer(net['voxres8_bn2'])
    net['voxres8_conv2'] = Conv3DDNNLayer(net['voxres8_relu2'], 128, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres8_out'] = ElemwiseSumLayer([net['conv7'], net['voxres8_conv2']])
    # VoxRes block 9
    net['voxres9_bn1'] = BatchNormLayer(net['voxres8_out'])
    net['voxres9_relu1'] = NonlinearityLayer(net['voxres9_bn1'])
    net['voxres9_conv1'] = Conv3DDNNLayer(net['voxres9_relu1'], 128, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres9_bn2'] = BatchNormLayer(net['voxres9_conv1'])
    net['voxres9_relu2'] = NonlinearityLayer(net['voxres9_bn2'])
    net['voxres9_conv2'] = Conv3DDNNLayer(net['voxres9_relu2'], 128, 3,
                                          pad='same', nonlinearity=identity)
    net['voxres9_out'] = ElemwiseSumLayer([net['voxres8_out'],
                                           net['voxres9_conv2']])

    net['pool10'] = Pool3DDNNLayer(net['voxres9_out'], 7)
    net['fc11'] = DenseLayer(net['pool10'], 128)
    net['prob'] = DenseLayer(net['fc11'], 2, nonlinearity=softmax)

    return net

# Logging setup
logging.basicConfig(format='[%(asctime)s]  %(message)s',
                    datefmt='%d.%m %H:%M:%S',
                    level=logging.DEBUG)


def run_training(first_class, second_class, results_folder,
                 num_epochs=70, batchsize=3):
    """Iterate minibatches on train subset.

    Parameters
    ----------
    first_class : {'AD', 'LMCI', 'EMCI', 'Normal'}
        String label for target == 0.
    second_class : {'AD', 'LMCI', 'EMCI', 'Normal'}
        String label for target == 1.
    results_folder : string
        Folder to store results.
    num_epochs : integer
        Number of epochs for all of the experiments. Default is 70.
    batchsize : integer
        Batchsize for network training. Default is 3.
    """

    if first_class not in {'AD', 'LMCI', 'EMCI', 'Normal'}:
        msg = "First class must be 'AD', 'LMCI', 'EMCI' or 'Normal', not {0}"
        raise ValueError(msg.format(first_class))

    if second_class not in {'AD', 'LMCI', 'EMCI', 'Normal'}:
        msg = "Second class must be 'AD', 'LMCI', 'EMCI' or 'Normal', not {0}"
        raise ValueError(msg.format(second_class))

    if first_class == second_class:
        raise ValueError("Class labels should be different")

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    metadata = pd.read_csv('data/metadata.csv')
    print (metadata)
    metadata.columns = ['Index','Label', 'Path']
    smc_mask = ((metadata.Label == first_class) | (metadata.Label == second_class)).values.astype('bool')
    data = np.zeros((smc_mask.sum(), 1, 110, 110, 110), dtype='float32')

    for it, im in tqdm(enumerate(metadata[smc_mask].Path.values),
                       total=smc_mask.sum(), desc='Reading MRI to memory'):
        mx = nib.load(im).get_data().max(axis=0).max(axis=0).max(axis=0)
        data[it, 0, :, :, :] = np.array(nib.load(im).get_data()) / mx

    target = (metadata[smc_mask].Label == second_class).values.astype('int32')

    for cvrand in range(5):
        cv = StratifiedKFold(target, n_folds=5, random_state=42 * cvrand,
                             shuffle=True)

        for fold, (train_index, test_index) in enumerate(cv):
            logging.debug('Starting fold {}'.format(fold))
            X_train, y_train = data[train_index], target[train_index]
            X_test, y_test = data[test_index], target[test_index]

            net = build_net()

            prediction = lasagne.layers.get_output(net['prob'])
            loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                               target_var)
            loss = loss.mean()

            params = lasagne.layers.get_all_params(net['prob'], trainable=True)
            updates = lasagne.updates.nesterov_momentum(loss, params, 0.001)

            test_prediction = lasagne.layers.get_output(net['prob'],
                                                        deterministic=True)
            test_loss = categorical_crossentropy(test_prediction, target_var)
            test_loss = test_loss.mean()

            train_fn = theano.function([input_var, target_var], loss,
                                       updates=updates)
            val_fn = theano.function([input_var, target_var], test_loss)
            test_fn = theano.function([input_var], test_prediction)

            logging.debug("Done building net")

            eps = []
            tr_losses = []
            val_losses = []
            val_accs = []
            val_rocs = []

            logging.debug("Starting training...")
            den = X_train.shape[0] / batchsize
            for epoch in range(num_epochs):
                train_err = 0
                train_batches = 0
                start_time = time.time()
                t = tqdm(iterate_minibatches_train(X_train, y_train, batchsize,
                                                   shuffle=True),
                         total=int(den),
                         desc='Epoch {}, Loss = inf'.format(epoch + 1))
                for batch in t:
                    inputs, targets = batch
                    train_err += train_fn(inputs, targets)
                    train_batches += 1
                    ls = train_err / train_batches
                    t.set_description('Epoch {}, Loss = {:.5f}'.format(epoch +
                                                                       1, ls))

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

                logging.debug("Epoch {} done".format(epoch + 1))
                print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, num_epochs, time.time() - start_time),
                    flush=True)
                print("  training loss:\t\t{:.7f}".format(
                    train_err / train_batches), flush=True)
                print("  validation loss:\t\t{:.7f}".format(
                    val_err / val_batches), flush=True)
                print("  validation accuracy:\t\t{:.7f}".format(
                    accuracy_score(np.array(targ),
                                   np.array(preds).argmax(axis=1))),
                    flush=True)
                print("  validation auc:\t\t{:.7f}".format(
                    roc_auc_score(np.array(targ),
                                  np.array(preds)[:, 1])), flush=True)

                eps.append(epoch)
                tr_losses.append(train_err / train_batches)
                val_losses.append(val_err / val_batches)
                val_accs.append(accuracy_score(np.array(targ),
                                               np.array(preds).argmax(axis=1)))
                val_rocs.append(roc_auc_score(np.array(targ),
                                              np.array(preds)[:, 1]))

            np.save('./{}/{}_{}nm_eps.npy'.format(results_folder,
                                                  cvrand, fold),
                    np.array(eps))
            np.save('./{}/{}_{}nm_tr_loss.npy'.format(results_folder,
                                                      cvrand, fold),
                    np.array(tr_losses))
            np.save('./{}/{}_{}nm_vl_loss.npy'.format(results_folder,
                                                      cvrand, fold),
                    np.array(val_losses))
            np.save('./{}/{}_{}nm_vl_accs.npy'.format(results_folder,
                                                      cvrand, fold),
                    np.array(val_accs))
            np.save('./{}/{}_{}nm_vl_rocs.npy'.format(results_folder,
                                                      cvrand, fold),
                    np.array(val_rocs))

run_training('AD', 'Normal', 'results_resnet/ad_vs_norm')

run_training('AD', 'EMCI', 'results_resnet/ad_vs_emci')

run_training('AD', 'LMCI', 'results_resnet/ad_vs_lmci')

run_training('EMCI', 'Normal', 'results_resnet/emci_vs_norm')

run_training('LMCI', 'Normal', 'results_resnet/lmci_vs_norm')

run_training('LMCI', 'EMCI', 'results_resnet/lmci_vs_emci')
