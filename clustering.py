import numpy
import sys
import os
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from utils import load_dataset, n_classes
from mimic_models import MimicBetaInitModelL2, MimicBetaInitModelConvL2

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

if len(sys.argv) > 2:
    oar_job_id = sys.argv[1]
    dataset = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3] == "Conv":
        model_class = MimicBetaInitModelConvL2
    else:
        model_class = MimicBetaInitModelL2
else:
    oar_job_id = "-1"
    dataset = "StarLightCurves"
    model_class = MimicBetaInitModelConvL2

nkiter = 500
ratio_n_shapelets = 10
model_path = "models/"
distances_path = "distances/"

if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(distances_path):
    os.makedirs(distances_path)

x_train, y_train, x_test, y_test = load_dataset(dataset)
data = numpy.concatenate((x_train, x_test))
n_ts, ts_len = data.shape[:2]
gt_labels = numpy.concatenate((y_train, y_test))
n_clusters = n_classes(gt_labels)

shapelet_lengths = {}
for sz in [int(p * ts_len) for p in [.15, .3, .45]]:
    n_shapelets = int(numpy.log(ts_len - sz) * ratio_n_shapelets)  # 2, 5, 8, 10
    shapelet_lengths[sz] = n_shapelets
print(dataset, shapelet_lengths, model_class.__name__)

m = model_class(shapelet_lengths, d=data.shape[2], print_loss_every=1000, ada_grad=True, niter=1000,
                print_approx_loss=True)
m.fit(data)
for ikiter in range(nkiter):
    m.partial_fit(data, 1000, (ikiter + 1) * 1000)
    model_fname = "%s%s_%dkiter_%s_%s.model" % (model_path, m.__class__.__name__, ikiter + 1, dataset, oar_job_id)
    m.dump_without_dists(model_fname)

model_fname = "%s%s_final_%s.model" % (model_path, m.__class__.__name__, dataset)
m.dump_without_dists(model_fname)
print("Saved model %s with approximate loss: %f (beta=%f)" % (model_fname, m._loss(data), m.beta))

data_shtr = numpy.empty((data.shape[0], sum(shapelet_lengths.values())))
for i in range(data.shape[0]):
    data_shtr[i] = m._shapelet_transform(data[i])

km = KMeans(n_clusters=n_clusters)
pred_labels = km.fit_predict(data_shtr)
print("Model=%s, dataset=%s, NMI(kmeans)=%f" % (m.__class__.__name__, dataset,
                                                normalized_mutual_info_score(gt_labels, pred_labels)))


