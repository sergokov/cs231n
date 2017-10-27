import random
import numpy as np
import time
import matplotlib.pyplot as plt

from utils import load_CIFAR10
from linear_classifier import LinearSVM


def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i].T
                dW[:, y.item(i)] += -X[i].T

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)
    correct_class_scores = scores[range(num_train), list(y)].reshape(-1, 1)  # (N, 1)
    margins = np.maximum(0, scores - correct_class_scores + 1)
    margins[range(num_train), list(y)] = 0
    loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)
    coeff_mat = np.zeros((num_train, num_classes))
    coeff_mat[margins > 0] = 1
    coeff_mat[range(num_train), list(y)] = 0
    coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)

    dW = (X.T).dot(coeff_mat)
    dW = dW / num_train + reg * W

    return loss, dW


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = \
        load_CIFAR10('/home/sergo/Work/ML_Experiments/cifar-10-batches-py/')

    train_data = np.reshape(train_data, (train_data.shape[0], -1))
    test_data = np.reshape(test_data, (test_data.shape[0], -1))

    mask = random.sample(xrange(0, len(train_data)), 100)

    dev_data = train_data[mask]
    dev_labels = train_labels[mask]

    mean_image = np.mean(train_data, axis=0)
    # plt.figure(figsize=(4, 4))
    # plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8'))
    # plt.show()

    train_data -= mean_image
    test_data -= mean_image
    dev_data -= mean_image

    train_data = np.hstack([train_data, np.ones((train_data.shape[0], 1))])
    test_data = np.hstack([test_data, np.ones((test_data.shape[0], 1))])
    dev_data = np.hstack([dev_data, np.ones((dev_data.shape[0], 1))])

    # generate a random SVM weight matrix of small numbers
    W = np.random.randn(3073, 10) * 0.0001

    tic = time.time()
    loss_naive, grad_naive = svm_loss_naive(W, dev_data, dev_labels, 0.00001)
    toc = time.time()
    print 'Naive loss: %e computed in %fs' % (loss_naive, toc - tic)

    svm = LinearSVM(W)
    tic = time.time()
    loss_vectorized, _ = svm.loss(dev_data, dev_labels, 0.00001)
    toc = time.time()
    print 'Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic)
