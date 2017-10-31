import random
import numpy as np
import time
import matplotlib.pyplot as plt

from utils import load_CIFAR10
from linear_classifier import LinearSVM


def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape)
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

    loss /= num_train
    dW /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


if __name__ == '__main__':
    train_data, train_labels, val_data, val_labels = \
        load_CIFAR10('/home/sergo/Work/ML_Experiments/cifar-10-batches-py/')

    train_data = np.reshape(train_data, (train_data.shape[0], -1))
    val_data = np.reshape(val_data, (val_data.shape[0], -1))

    mask = random.sample(xrange(0, len(train_data)), 100)

    dev_data = train_data[mask]
    dev_labels = train_labels[mask]

    mean_image = np.mean(train_data, axis=0)
    # plt.figure(figsize=(4, 4))
    # plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8'))
    # plt.show()

    train_data -= mean_image
    val_data -= mean_image
    dev_data -= mean_image

    train_data = np.hstack([train_data, np.ones((train_data.shape[0], 1))])
    val_data = np.hstack([val_data, np.ones((val_data.shape[0], 1))])
    dev_data = np.hstack([dev_data, np.ones((dev_data.shape[0], 1))])

    # # generate a random SVM weight matrix of small numbers
    # W = np.random.randn(3073, 10) * 0.0001
    #
    # tic = time.time()
    # loss_naive, grad_naive = svm_loss_naive(W, dev_data, dev_labels, 0.00001)
    # toc = time.time()
    # print 'Naive loss: %e computed in %fs' % (loss_naive, toc - tic)
    #
    # svm = LinearSVM(W)
    # tic = time.time()
    # loss_vectorized, _ = svm.loss(dev_data, dev_labels, 0.00001)
    # toc = time.time()
    # print 'Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic)
    #
    #
    # print 'Start training process...'
    # svm = LinearSVM()
    # loss_hist = svm.train(train_data, train_labels, learning_rate=1e-7, num_iters=1500, verbose=False)
    # print 'End training process.'
    #
    # plt.plot(loss_hist)
    # plt.xlabel('Ittereation Number')
    # plt.ylabel('Loss value')
    # plt.show(block=True)
    #
    # val_pred = svm.predict(val_data)
    # print 'Accuracy: %f' % np.mean(val_labels == val_pred)

    learning_rates = [1.4e-7, 1.5e-7, 1.6e-7]
    range = range(-3, 3)
    regularization_strengths = [(1 + i * 0.1) * 1e4 for i in range] + [(2 + 0.1 * i) * 1e4 for i in range]

    results = {}
    best_val = -1
    best_svm = None

    for rs in regularization_strengths:
        for lr in learning_rates:
            svm = LinearSVM()
            loss_hist = svm.train(train_data, train_labels, lr, rs, num_iters=3000)
            train_labels_pred = svm.predict(train_data)
            train_accuracy = np.mean(train_labels == train_labels_pred)
            val_labels_pred = svm.predict(val_data)
            val_accuracy = np.mean(val_labels == val_labels_pred)
            if val_accuracy > best_val:
                best_val = val_accuracy
                best_svm = svm
            results[(lr, rs)] = train_accuracy, val_accuracy
            print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, rs, train_accuracy, val_accuracy)

    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy)

    print 'Best cross-validation accuracy: %f' % best_val

    import math
    x_scatter = [math.log10(x[0]) for x in results]
    y_scatter = [math.log10(x[1]) for x in results]

    marker_size = 100
    colors = [results[x][0] for x in results]
    plt.subplot(2, 1, 1)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 training accuracy')

    colors = [results[x][1] for x in results]  # default size of markers is 20
    plt.subplot(2, 1, 2)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 validation accuracy')
    plt.show(block=True)

    w = best_svm.W[:-1, :]
    w = w.reshape(32, 32, 3, 10)
    w_min, w_max = np.min(w), np.max(w)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in xrange(10):
        plt.subplot(2, 5, i + 1)
        wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
    plt.show()
