import numpy as np


class LinearClassifier(object):
    def __init__(self, W = None):
        self.W = W

    def train(self, X, y, learning_rate=1e-6, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []
        for it in xrange(num_iters):
            batch_idx = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W += - learning_rate * grad
            if verbose and it % 100 == 0:
               print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

        return loss_history

    def predict(self, X):
        return np.argmax(X.dot(self.W), axis=1)

    def loss(self, X, y, reg):
        pass


class LinearSVM(LinearClassifier):

  def loss(self, X, y, reg):
      num_train = X.shape[0]
      num_classes = self.W.shape[1]
      scores = X.dot(self.W)
      correct_class_scores = scores[range(num_train), list(y)].reshape(-1, 1)
      margins = np.maximum(0, scores - correct_class_scores + 1)
      margins[range(num_train), list(y)] = 0
      loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(self.W * self.W)
      coeff_mat = np.zeros((num_train, num_classes))
      coeff_mat[margins > 0] = 1
      coeff_mat[range(num_train), list(y)] = 0
      coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)

      dW = (X.T).dot(coeff_mat)
      dW = dW / num_train + reg * self.W

      return loss, dW


class Softmax(LinearClassifier):

  def loss(self, X, y, reg):
      loss = 0.0
      dW = np.zeros_like(self.W)
      num_classes = self.W.shape[1]
      num_train = X.shape[0]
      scores = X.dot(self.W)
      shift_scores = scores - np.max(scores, axis=1).reshape(-1, 1)
      softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1).reshape(-1, 1)
      loss = -np.sum(np.log(softmax_output[range(num_train), list(y)]))
      loss /= num_train
      loss += 0.5 * reg * np.sum(self.W * self.W)
      dS = softmax_output.copy()
      dS[range(num_train), list(y)] += -1
      dW = (X.T).dot(dS)
      dW = dW / num_train + reg * self.W

      return loss, dW