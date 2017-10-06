import random
import numpy as np
from utils import load_CIFAR10
import matplotlib.pyplot as plt


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
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i].T
                dW[:, y[i]] += -X[i].T

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = \
        load_CIFAR10('/home/sergo/Work/ML_Experiments/cifar-10-batches-py/')

    train_data = np.reshape(train_data, (train_data.shape[0], -1))
    train_labels = np.reshape(train_labels, (train_labels.shape[0], -1))
    test_data = np.reshape(test_data, (test_data.shape[0], -1))
    test_labels = np.reshape(test_labels, (test_labels.shape[0], -1))

    mean_image = np.mean(train_data, axis=0)
    # plt.figure(figsize=(4, 4))
    # plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8'))
    # plt.show()

    train_data -= mean_image
    test_data -= mean_image

    train_data = np.hstack([train_data, np.ones((train_data.shape[0], 1))])
    test_data = np.hstack([test_data, np.ones((test_data.shape[0], 1))])

    # generate a random SVM weight matrix of small numbers
    W = np.random.randn(3073, 10) * 0.0001

    loss, grad = svm_loss_naive(W, test_data, test_labels, 0.00001)
    print 'loss: %f' % (loss,)
