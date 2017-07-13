import numpy

from mimic_models import MimicModelIncremental
from utils import load_dataset

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

x_train, y_train, x_test, y_test = load_dataset("Adiac")  # Adiac data must be in "data/ucr/Adiac" folder
data = numpy.concatenate((x_train, x_test))


# 50 shapelets of size 10, 50 of size 20
# 5 shapelets per group and per size
m = MimicModelIncremental(shapelet_lengths={10: 50, 20: 50}, size_shapelet_groups=5)

m.fit(data)
