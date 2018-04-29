
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm
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
import theano.tensor as T
import pickle
from skimage.transform import rotate
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[ ]:


input_var = T.tensor5(name='input', dtype='float32')
target_var = T.ivector()


# ### Function for heatmap calculation

# In[ ]:


def occlusion_heatmap(net, x, label, square_size=7, batchsize=1):
    """
    Parameters
    ----------
    net : Lasagne Layer
        The neural net to test.
    x : np.array
        The input data, should be of shape (1, c, x, y, z).
    label : int
        Sample label
    square_size : int, optional (default=7)
        The length of the side of the square that occludes the image.
    batchsize : int, optional (default=1)
        Number of images in batch for inference pass.
        
    Results
    -------
    np.array
        3D np.array that at each point (i, j) contains the predicted
        probability of the correct class if the image is occluded by a
        square with center (i, j).
    """
    if (x.ndim != 5) or x.shape[0] != 1:
        raise ValueError("This function requires the input data to be of shape"
                         " (1, c, x, y, z), instead got {}".format(x.shape))
    if square_size % 2 == 0:
        raise ValueError("Square length has to be an odd number, instead "
                         "got {}.".format(square_size))

    num_classes = 2
    img = x[0].copy()
    bs, col, s0, s1, s2 = x.shape

    pad = square_size // 2 + 1
    x_occluded = np.zeros((s2, col, s0, s1, s2), dtype=img.dtype)
    probs = np.zeros((s0, s1, s2, num_classes))

    for i in range(s0):
        for j in tqdm(range(s1), desc='x = {}'.format(i)):
            batch_iter = 1
            batch = []
            for k in range(s2):
                x_pad = np.pad(img, ((0, 0), (pad, pad),
                                     (pad, pad), (pad, pad)), 'constant')
                x_pad[:,
                      i:i + square_size,
                      j:j + square_size,
                      k:k + square_size] = 0.
                x_occluded = x_pad[:, pad:-pad, pad:-pad, pad:-pad]
                batch.append(x_occluded)
                if batch_iter % batchsize == 0:
                    y_proba = test_fn(np.array(batch).reshape((-1, 1, 110, 110, 110)))
                    probs[i, j, k - batchsize + 1:k + 1, :] = y_proba
                    batch_iter = 0
                    batch = []
                batch_iter += 1

    return probs


# In[ ]:


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


# ### Network initialization with pretrained weights

# In[ ]:


net = build_net()
test_prediction = lasagne.layers.get_output(net['prob'], deterministic=True)
test_fn = theano.function([input_var], test_prediction)

with open('data/resnet_weights.pkl', 'rb') as f:
    weights = pickle.load(f)
    lasagne.layers.set_all_param_values(net['prob'], weights)


# ### Load data and split into the same validation folds

# In[ ]:


metadata = pd.read_csv('data/metadata.csv')
smc_mask = ((metadata.Label == 'Normal') | (
    metadata.Label == 'AD')).values.astype('bool')
data = np.zeros((smc_mask.sum(), 1, 110, 110, 110), dtype='float32')

for it, im in tqdm(enumerate(metadata[smc_mask].Path.values),
                   total=smc_mask.sum(), desc='Reading MRI to memory'):
    mx = nib.load(im).get_data().max(axis=0).max(axis=0).max(axis=0)
    data[it, 0, :, :, :] = np.array(nib.load(im).get_data()) / mx

target = (metadata[smc_mask].Label != 'Normal').values.astype('int32')

cv = StratifiedKFold(target, n_folds=5, random_state=0, shuffle=True)


# ### Select sample for heatmap plot

# In[ ]:


for fold, (train_index, test_index) in enumerate(cv):
    X_train, y_train = data[train_index], target[train_index]
    X_test, y_test = data[test_index], target[test_index]

    for it, img in enumerate(X_test):
        print(test_fn(img.reshape((1, 1, 110, 110, 110))).reshape(-1,),
              y_test[it])
    break


# In[ ]:


res = occlusion_heatmap(net, X_test[2].reshape(1, 1, 110, 110, 110), 0)


# ### Plot heatmap as overlay

# In[ ]:


plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.axis('off')
matr = rotate(X_test[2][:, :, :, 47].reshape(110, 110), 90)
plt.imshow(matr, cmap=plt.cm.Greys_r, interpolation=None,
           vmax=1., vmin=0.)
plt.hold(True)
matr = rotate(res[:, :, 47, 0].reshape(110, 110), 90)
plt.imshow(200. * (1 - np.ma.masked_where(matr > .999, matr)),
           interpolation=None, vmax=1., vmin=.0, alpha=.8,
           cmap=plt.cm.viridis_r)
plt.subplot(1, 2, 2)
plt.axis('off')
matr = rotate(X_test[2][:, 57, :, :].reshape(110, 110), 90)
plt.imshow(matr, cmap=plt.cm.Greys_r, interpolation=None,
           vmax=1., vmin=0.)
plt.hold(True)
matr = rotate(res[57, :, :, 0].reshape(110, 110), 90)
plt.imshow(200. * (1 - np.ma.masked_where(matr > .999, matr)),
           interpolation=None, vmax=1., vmin=.0, alpha=.8,
           cmap=plt.cm.viridis_r)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=20)
plt.show()

