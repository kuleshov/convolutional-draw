import sys
import os
import pickle
import tarfile
import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

# ----------------------------------------------------------------------------

def load_cifar10():
  """Download and extract the tarball from Alex's website."""
  dest_directory = '.'
  DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    if sys.version_info[0] == 2:
      from urllib import urlretrieve
    else:
      from urllib.request import urlretrieve

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)  

  def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
      datadict = pickle.load(f)
      X = datadict['data']
      Y = datadict['labels']
      X = X.reshape(10000, 3, 32, 32,).astype("float32")
      X = np.transpose(X, [0, 2, 3, 1])
      Y = np.array(Y, dtype=np.uint8)
      return X, Y

  xs, ys = [], []
  for b in range(1,6):
    f = 'cifar-10-batches-py/data_batch_%d' % b
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch('cifar-10-batches-py/test_batch')
  return Xtr, Ytr, Xte, Yte

def load_svhn():
  if sys.version_info[0] == 2:
    from urllib import urlretrieve
  else:
    from urllib.request import urlretrieve

  def download(filename, source="https://github.com/smlaine2/tempens/raw/master/data/svhn/"):
    print "Downloading %s" % filename
    urlretrieve(source + filename, filename)

  import cPickle
  def load_svhn_files(filenames):
    if isinstance(filenames, str):
        filenames = [filenames]
    images = []
    labels = []
    for fn in filenames:
        if not os.path.isfile(fn): download(fn)
        with open(fn, 'rb') as f:
          X, y = cPickle.load(f)
        images.append(np.asarray(X, dtype='float32') / np.float32(255))
        labels.append(np.asarray(y, dtype='uint8'))
    return np.concatenate(images), np.concatenate(labels)

  X_train, y_train = load_svhn_files(['train_%d.pkl' % i for i in (1, 2, 3)])
  X_test, y_test = load_svhn_files('test.pkl')

  return X_train, y_train, X_test, y_test  

# ----------------------------------------------------------------------------

class Dataset(object):

  def __init__(self,
               datapoints,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)

    if labels is None:
      labels = np.zeros((len(datapoints),))

    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert datapoints.shape[0] == labels.shape[0], (
          'datapoints.shape: %s labels.shape: %s' % (datapoints.shape, labels.shape))
      self._num_examples = datapoints.shape[0]

    self._datapoints = datapoints
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def datapoints(self):
    return self._datapoints

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._datapoints = self.datapoints[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      datapoints_rest_part = self._datapoints[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._datapoints = self.datapoints[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      datapoints_new_part = self._datapoints[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((datapoints_rest_part, datapoints_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._datapoints[start:end], self._labels[start:end]
