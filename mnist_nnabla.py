import os
import argparse
import time
import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource

from keras.datasets import mnist

class MnistDataSource(DataSource):

  def _get_data(self, position):
    image = self._images[self._indexes[position]]
    label = self._labels[self._indexes[position]]
    return (image, label)

  def __init__(self, x, y, shuffle=False, rng=None):
    super(MnistDataSource, self).__init__(shuffle=shuffle)

    self._images = x
    self._labels = y

    self._size = self._labels.size
    self._variables = ('x', 'y')
    if rng is None:
      rng = np.random.RandomState(313)
    self.rng = rng
    self.reset()

  def reset(self):
    if self._shuffle:
      self._indexes = self.rng.permutation(self._size)
    else:
      self._indexes = np.arange(self._size)
    super(MnistDataSource, self).reset()

  @property
  def images(self):
    """Get copy of whole data with a shape of (N, 1, H, W)."""
    return self._images.copy()

  @property
  def labels(self):
    """Get copy of whole label with a shape of (N, 1)."""
    return self._labels.copy()

def network(image):
  c1 = PF.convolution(image, 32, kernel=(5, 5), name='conv1', pad=(2, 2))
  c1 = F.max_pooling(F.relu(c1, inplace=True), kernel=(2, 2))
  c2 = PF.convolution(c1, 64, kernel=(5, 5), name='conv2', pad=(2, 2))
  c2 = F.max_pooling(F.relu(c2, inplace=True), kernel=(2, 2))
  c3 = F.relu(PF.affine(c2, n_outmaps=512, name='fc3'), inplace=True)
  c4 = F.dropout(c3, p=0.5)
  c4 = F.softmax(PF.affine(c3, n_outmaps=10, name='fc4'))
  return c4

def main(verbose):
  (x_train, y_train), _ = mnist.load_data()
  x_train = x_train / 255.0
  x_train = x_train[..., None]
  x_train = x_train.transpose([0, 3, 1, 2])
  x_train = x_train.astype(np.float32)
  x_train = np.ascontiguousarray(x_train)
  y_train = y_train.astype(np.int64)
  y_train = y_train[:, None]

  batch_size = 32
  dataset = MnistDataSource(x_train, y_train, shuffle=True)
  loader = data_iterator(dataset, batch_size)

  # TRAIN
  # Create input variables.
  image = nn.Variable([batch_size, 1, 28, 28])
  label = nn.Variable([batch_size, 1])
  # Create prediction graph.
  pred = network(image)
  pred.persistent = True
  # Create loss function.
  loss = F.categorical_cross_entropy(pred, label)

  # Create Solver.
  solver = S.Adam(0.001)
  solver.set_parameters(nn.get_parameters())

  for epoch in range(2):
    if epoch == 1:
      start = time.time()
    for i, data in enumerate(loader):
      if loader.epoch != epoch :
        break
      # Training forward
      image.d = data[0]
      label.d = data[1]
      solver.zero_grad()
      loss.forward(clear_no_need_grad=True)
      loss.backward(clear_buffer=True)
      solver.update()
      if verbose:
        print('{i}: {loss}'.format(i=i, loss=loss.d), end=" "*16+"\r")
    if epoch == 1:
      print('Elapsed time: {t}'.format(t=time.time()-start))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='MNIST')
  parser.add_argument('--use-cuda', action='store_true', default=False, help='Use CUDA')
  parser.add_argument('--verbose', action='store_true', default=False, help='verbose')
  args = parser.parse_args()

  # Get context.
  from nnabla.contrib.context import extension_context
  extension_module = 'cpu'
  device_id = 0
  if args.use_cuda:
    extension_module = 'gpu'
  ctx = extension_context(extension_module, device_id=device_id)
  nn.set_default_context(ctx)

  main(args.verbose)
