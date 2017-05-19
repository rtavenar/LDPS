import numpy
import os
from scipy.spatial.distance import cdist

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def load_shapelets_1d(fname):  # TODO: deal with shapelets having different lengths
    shapelets = numpy.loadtxt(fname)
    K, L = shapelets.shape
    return shapelets.reshape((K, L, 1))


def load_dataset(ds_name, path=None):
    if path is None:
        path = "data/ucr"
    directory = "%s/%s" % (path, ds_name)
    fname_train, fname_test = None, None
    for fname in os.listdir(directory):
        if fname.endswith("_TRAIN.txt"):
            fname_train = os.path.join(directory, fname)
        elif fname.endswith("_TEST.txt"):
            fname_test = os.path.join(directory, fname)
    data_train = numpy.loadtxt(fname_train)
    n, sz = data_train.shape
    x_train, y_train = data_train[:, 1:].reshape((n, sz - 1, 1)), data_train[:, 0]
    data_test = numpy.loadtxt(fname_test)
    n, sz = data_test.shape
    x_test, y_test = data_test[:, 1:].reshape((n, sz - 1, 1)), data_test[:, 0]
    return x_train, y_train, x_test, y_test


def dtw_sq(s1, s2):
    cross_dist = cdist(s1.reshape((-1, 1)), s2.reshape((-1, 1)), "sqeuclidean")
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = numpy.zeros((l1 + 1, l2 + 1))
    cum_sum[1:, 0] = numpy.inf
    cum_sum[0, 1:] = numpy.inf
    for i in range(l1):
        for j in range(l2):
            if numpy.isfinite(cum_sum[i + 1, j + 1]):
                pred_list = [cum_sum[i, j + 1], cum_sum[i + 1, j], cum_sum[i, j]]
                argmin_pred = numpy.argmin(pred_list)
                cum_sum[i + 1, j + 1] = pred_list[argmin_pred] + cross_dist[i, j]
    return cum_sum[-1, -1]


def dtw(s1, s2):
    return numpy.sqrt(dtw_sq(s1, s2))


def dtw_sq_normalized(s1, s2):
    sz = max(s1.shape[0], s2.shape[0])
    return dtw_sq(s1, s2) / (s1.shape[0] + s2.shape[0])


def n_classes(y):
    return len(list(set(y.astype(numpy.int32))))
