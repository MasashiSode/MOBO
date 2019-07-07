import numpy as np


def ucb(y_pred, y_train):
    mean = y_pred.mean.numpy()[0],
    std = y_pred.stddev.numpy()[0],
    n_sample = y_train.numpy().shape[0]
    # z = (mean - y_min) / std
    # out = (mean - y_min) * norm.cdf(z) + std * norm.pdf(z)
    out = mean[0] + (np.sqrt(np.log(n_sample) / n_sample)) * std[0]

    return out
