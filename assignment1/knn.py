import cPickle as pickle
import numpy as np
import os


def load_CIFAR_batch(filename):
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y


def load_CIFAR10(ROOT):
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte


class KNearestNeighbor(object):
    def __init__(self):
        self.train_data = None
        self.train_labels = None

    def train(self, X, Y):
        self.train_data = np.array(X)
        self.train_labels = np.array(Y)

    def predict(self, X, k=1):
        num_test = X.shape[0]
        pred = np.zeros(num_test)
        for i in xrange(num_test):
            sum = np.sum(np.abs(self.train_data - X[i]), axis=1)
            k_closest = self.train_labels[sum.argsort()[:k]]
            pred[i] = np.argmax(np.bincount(k_closest))
        return pred


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = load_CIFAR10('/home/sergo/Work/ML_Experiments/cifar-10-batches-py/')
    train_data = train_data.reshape(train_data.shape[0], 32 * 32 * 3)[:5000]
    train_labels = train_labels[:5000]
    test_data = test_data.reshape(test_data.shape[0], 32 * 32 * 3)[:500]
    test_labels = test_labels[:500]
    knn = KNearestNeighbor()
    knn.train(train_data, train_labels)
    pred = knn.predict(test_data, k=10)
    print "accuracy: %s" % (np.mean(pred == test_labels))