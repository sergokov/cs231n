import random
import numpy as np
import time
import matplotlib.pyplot as plt

from utils import load_CIFAR10
from linear_classifier import LinearSVM

if __name__ == '__main__':
    train_data, train_labels, val_data, val_labels = \
        load_CIFAR10('/home/sergo/Work/ML_Experiments/Data/cifar-10-batches-py/')

    train_data = np.reshape(train_data, (train_data.shape[0], -1))
    val_data = np.reshape(val_data, (val_data.shape[0], -1))

    mask = random.sample(xrange(0, len(train_data)), 100)

    dev_data = train_data[mask]
    dev_labels = train_labels[mask]

    mean_image = np.mean(train_data, axis=0)
    plt.figure(figsize=(4, 4))
    plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8'))
    plt.show()

    train_data -= mean_image
    val_data -= mean_image
    dev_data -= mean_image

    train_data = np.hstack([train_data, np.ones((train_data.shape[0], 1))])
    val_data = np.hstack([val_data, np.ones((val_data.shape[0], 1))])
    dev_data = np.hstack([dev_data, np.ones((dev_data.shape[0], 1))])
