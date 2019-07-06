from scipy.stats import norm


def ei(y_pred, y_train):
    mean = y_pred.mean.numpy()[0],
    std = y_pred.stddev.numpy()[0],
    y_min = y_train.numpy().min()
    z = (mean - y_min) / std
    out = (mean - y_min) * norm.cdf(z) + std * norm.pdf(z)
    return out[0]
