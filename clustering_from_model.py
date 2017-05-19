import numpy
import sys
import os
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from utils import load_dataset, n_classes
from mimic_models import MimicModelL2

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

use_dists = True
dataset = sys.argv[1]
fname_model = sys.argv[2]

x_train, y_train, x_test, y_test = load_dataset(dataset)
data = numpy.concatenate((x_train, x_test))
n_ts, ts_len = data.shape[:2]
gt_labels = numpy.concatenate((y_train, y_test))
n_clusters = n_classes(gt_labels)

m = MimicModelL2.load(fname_model)
shapelet_lengths = m.shapelet_lengths

data_shtr = numpy.empty((data.shape[0], sum(shapelet_lengths.values())))
for i in range(data.shape[0]):
    data_shtr[i] = m._shapelet_transform(data[i])

nmis = []
for i_trial in range(20):
    km = KMeans(n_clusters=n_clusters, random_state=i_trial)
    pred_labels = km.fit_predict(data_shtr)
    nmis.append(normalized_mutual_info_score(gt_labels, pred_labels))
    print("[%d] Model=%s, nmi:%f,ri:NA" % (i_trial, os.path.basename(fname_model), nmis[-1]))

print("Model=%s, dataset=%s, Avg. NMI(kmeans)=%f (+/- %f)" % (os.path.basename(fname_model), dataset, numpy.mean(nmis),
                                                              numpy.std(nmis)))
