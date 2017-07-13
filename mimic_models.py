import numpy
import sys
from sklearn.cluster import KMeans
import pickle
from numpy.lib.stride_tricks import as_strided
try:
    from cydtw import dtw
except:
    from utils import dtw

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class MimicModelL2:
    # shapelet_length is a dictionary with keys equal to shapelet length and values equal to number of such shapelets
    def __init__(self, shapelet_lengths, d=1, metric=None, convergence_rate=.01, niter=100000, init_method="kmeans",
                 ada_grad=False, print_loss_every=1000, print_approx_loss=False):
        self.n_shapelets = sum(shapelet_lengths.values())
        self.shapelet_lengths = shapelet_lengths
        self.d = d
        self.convergence_rate = convergence_rate
        self.init_method = init_method
        self.print_loss_every = print_loss_every
        self.print_approx_loss = print_approx_loss
        self.niter = niter
        self.ada_grad = ada_grad
        if metric is None:
            self.metric = dtw
        else:
            self.metric = metric
        self.beta = 1.
        self.shapelets = []
        self.past_gradients_Skl = []
        self.past_gradients_Beta = 0.
        self.precomputed_dists = {}

    @staticmethod
    def load(fname):
        m = pickle.load(open(fname, "rb"))
        if m.precomputed_dists is None:
            m.precomputed_dists = {}
        return m

    @staticmethod
    def load_with_dists(fname, fname_dists):
        m = pickle.load(open(fname, "rb"))
        if m.precomputed_dists is None:
            m.precomputed_dists = {}
        if fname_dists.endswith(".npy"):
            m.preload_distances_npy(fname_dists)
        else:
            m.preload_distances(fname_dists)
        return m

    def dump(self, fname):
        pickle.dump(self, open(fname, "wb"))

    def dump_without_dists(self, fname):
        pc = self.precomputed_dists
        self.precomputed_dists = {}
        pickle.dump(self, open(fname, "wb"))
        self.precomputed_dists = pc

    def preload_distances(self, fname, read_binary=False):
        if read_binary:
            mat = numpy.load(fname)
        else:
            mat = numpy.loadtxt(fname)
        assert mat.shape[0] == mat.shape[1]
        self.precomputed_dists = {}
        for i in range(mat.shape[0]):
            for j in range(i + 1, mat.shape[1]):
                if mat[i, j] >= 0.:
                    self.precomputed_dists[i, j] = mat[i, j]

    def preload_distances_npy(self, fname):
        self.preload_distances(fname=fname, read_binary=True)

    def dump_distances(self, fname, n_ts=-1, write_binary=False):
        if n_ts < 0:
            max_i = max([k[0] for k in self.precomputed_dists.keys()])
            max_j = max([k[1] for k in self.precomputed_dists.keys()])
            n_ts = max(max_i, max_j) + 1
        mat = numpy.zeros((n_ts, n_ts)) - 1.
        for (i, j), dist in self.precomputed_dists.items():
            mat[i, j] = dist
        if write_binary:
            numpy.save(arr=mat, file=fname)
        else:
            numpy.savetxt(X=mat, fname=fname)

    def dump_distances_npy(self, fname, n_ts=-1):
        self.dump_distances(fname=fname, n_ts=n_ts, write_binary=True)

    def partial_fit(self, X, niter=None, idx_iter_start=0):
        if niter is None:
            niter = self.niter
        _X = X.reshape((X.shape[0], -1, self.d))
        for idx_iter in range(idx_iter_start, idx_iter_start + niter):
            self._update_shapelets_one_iter(_X, idx_iter)
            avg_dist = numpy.mean(list(self.precomputed_dists.values()))
            if (idx_iter + 1) % self.print_loss_every == 0:
                if self.print_approx_loss:
                    loss = self._approximate_loss(_X)
                else:
                    loss = self._loss(_X)
                print("iteration %d, loss=%f (average dist: %f, beta=%f)" % (idx_iter + 1, loss, avg_dist, self.beta))

    def fit(self, X):
        _X = X.reshape((X.shape[0], -1, self.d))
        self._init_shapelets(_X)
        self.partial_fit(_X)

    def finite_model(self):
        for shp in self.shapelets:
            if not numpy.alltrue(numpy.isfinite(shp)):
                return False
        return True

    def _loss(self, X):
        l = 0.
        n_ts, ts_len = X.shape[:2]
        transforms = numpy.array([self._shapelet_transform(Xi) for Xi in X])
        for i in range(n_ts):
            for j in range(i + 1, n_ts):
                l += (self.get_dist(X, i, j) - self._compute_approximate_dist_from_transforms(transforms[i],
                                                                                                    transforms[j])) ** 2
        n_pairs = n_ts * (n_ts - 1) / 2
        return l / (2 * n_pairs)

    def _approximate_loss(self, X, n_pairs=1000):
        return self._approximate_loss_and_dists(X, n_pairs=n_pairs)[0]

    def _approximate_loss_and_dists(self, X, n_pairs=1000):
        l = 0.
        n_ts, ts_len = X.shape[:2]
        random_indices = numpy.random.choice(n_ts, size=n_pairs * 2, replace=True).reshape((n_pairs, 2))
        dists = []
        for i, j in random_indices:
            i, j = min(i, j), max(i, j)
            dists.append(self.get_dist(X, i, j))
            l += (self.precomputed_dists[i, j] - self._compute_approximate_dist(X[i], X[j])) ** 2
        return l / (2 * n_pairs), numpy.mean(dists)

    def _save_fit(self, X, niter, loss):
        n_ts, ts_len = X.shape[:2]
        n_pairs = int(n_ts * (n_ts - 1) / 2)
        transforms = numpy.array([self._shapelet_transform(Xi) for Xi in X])
        dists = numpy.zeros((n_pairs, 2))
        idx = 0
        for i in range(n_ts):
            for j in range(i + 1, n_ts):
                dists[idx] = (self.get_dist(X, i, j),
                              self._compute_approximate_dist_from_transforms(transforms[i], transforms[j]))
                idx += 1
        import pylab
        import time
        pylab.figure()
        pylab.scatter(dists[:, 0], dists[:, 1])
        pylab.plot([numpy.min(dists), numpy.max(dists)], [numpy.min(dists), numpy.max(dists)], "r-", linewidth=2)
        pylab.title("Loss: %f" % loss)
        pylab.savefig("fig/fit_%s_niter%d_%d.pdf" % (self.__class__.__name__, niter, int(time.time())))
        pylab.close()

    def _init_shapelets(self, X):
        if self.init_method == "random":
            for shp_len in sorted(self.shapelet_lengths.keys()):
                for _ in range(self.shapelet_lengths[shp_len]):
                    self.shapelets.append(numpy.random.randn(shp_len, self.d))
        elif self.init_method == "kmeans":
            n_draw = 10000
            n_ts, sz, d = X.shape
            for shp_len in sorted(self.shapelet_lengths.keys()):
                indices_ts = numpy.random.choice(n_ts, size=n_draw, replace=True)
                indices_time = numpy.random.choice(sz - shp_len + 1, size=n_draw, replace=True)
                subseries = numpy.zeros((n_draw, shp_len, d))
                for i in range(n_draw):
                    subseries[i] = X[indices_ts[i], indices_time[i]:indices_time[i] + shp_len]
                subseries = subseries.reshape((n_draw, shp_len * d))
                shapelets = KMeans(n_clusters=self.shapelet_lengths[shp_len]).fit(subseries).cluster_centers_
                shapelets = shapelets.reshape((self.shapelet_lengths[shp_len], shp_len, d))
                for shp in shapelets:
                    self.shapelets.append(shp)
        else:
            raise NotImplementedError("Could not initialize shapelets: unknown method %s", self.init_method)

        if self.ada_grad:
            for shp_len in sorted(self.shapelet_lengths.keys()):
                for _ in range(self.shapelet_lengths[shp_len]):
                    self.past_gradients_Skl.append(numpy.zeros((shp_len, self.d)))
            self.past_gradients_Beta = 0.

    def _update_shapelets_one_iter(self, X, iter):
        grad_S, grad_Beta = self._grad(X)
        if self.ada_grad:
            for k in range(self.n_shapelets):
                conv_rate_scaling = numpy.ones_like(self.past_gradients_Skl[k])
                ind = self.past_gradients_Skl[k] > 1e-9
                conv_rate_scaling[ind] = numpy.sqrt(self.past_gradients_Skl[k][ind])
                self.shapelets[k] -= self.convergence_rate * grad_S[k] / conv_rate_scaling
            conv_rate_scaling = 1.
            if self.past_gradients_Beta > 1e-9:
                conv_rate_scaling = numpy.sqrt(self.past_gradients_Beta)
            new_beta = self.beta - self.convergence_rate * grad_Beta / conv_rate_scaling
            conv_rate_beta = self.convergence_rate / conv_rate_scaling
        else:
            for k in range(self.n_shapelets):
                self.shapelets[k] -= self.convergence_rate * grad_S[k]
            new_beta = self.beta - self.convergence_rate * grad_Beta
            conv_rate_beta = self.convergence_rate
        if new_beta < 0.:
            grad_Beta = self.beta / conv_rate_beta
            new_beta = 0.
        self.beta = new_beta
        if self.ada_grad:
            for k in range(self.n_shapelets):
                self.past_gradients_Skl[k] += grad_S[k] ** 2
            self.past_gradients_Beta += grad_Beta ** 2

    def _grad(self, X, i=None, j=None):
        if i is None and j is None:
            i, j = numpy.random.choice(X.shape[0], size=2, replace=False)
        elif i is None:
            i = numpy.random.choice(X.shape[0])
        elif j is None:
            j = numpy.random.choice(X.shape[0])
        i, j = min(i, j), max(i, j)
        y_target = self.get_dist(X, i, j)
        y_hat_no_beta = self._compute_approximate_dist_no_beta(X[i], X[j])
        y_hat = y_hat_no_beta * self.beta
        dY_ddelta = self._dY_ddelta(X[i], X[j])
        dL_dS = []
        for k in range(self.n_shapelets):
            sz = self.shapelets[k].shape[0]
            ti = self._idx_match(X[i], k)
            tj = self._idx_match(X[j], k)
            dL_dSkl = (y_hat - y_target) * dY_ddelta[k] * self._dM_dSkl(X[i], ti, X[j], tj, sz) / 2
            dL_dS.append(dL_dSkl)
        dL_dBeta = y_hat_no_beta * (y_hat - y_target)
        return dL_dS, dL_dBeta

    def _dY_ddelta(self, Xi, Xj):
        delta = self._shapelet_transform(Xi) - self._shapelet_transform(Xj)
        delta_norm2 = numpy.linalg.norm(delta, ord=2)
        if delta_norm2 < 1e-10:
            sys.stderr.write("Problem when computing gradient of L2-norm: gradient set to zero for safety reasons\n")
            return numpy.zeros(delta.shape)
        else:
            return self.beta * delta / delta_norm2

    def _dM_dSkl(self, Xi, ti, Xj, tj, sz):
        return 2 * (Xj[tj:tj+sz] - Xi[ti:ti+sz]) / sz

    def _idx_match(self, Xi, k):
        shp = self.shapelets[k]
        assert Xi.shape[1] == shp.shape[1] == 1
        Xi = Xi.reshape((-1, ))
        shp = shp.reshape((-1, ))
        sz = shp.shape[0]
        elem_size = Xi.strides[0]
        Xi_reshaped = as_strided(Xi, strides=(elem_size, elem_size), shape=(Xi.shape[0] - sz + 1, sz))
        distances = numpy.linalg.norm(Xi_reshaped - shp, axis=1) ** 2
        return numpy.argmin(distances)

    def _shapelet_transform(self, Xi):
        ret = numpy.empty((self.n_shapelets, ))
        for k in range(self.n_shapelets):
            shp = self.shapelets[k]
            sz = shp.shape[0]
            ti = self._idx_match(Xi, k)
            ret[k] = numpy.linalg.norm(Xi[ti:ti+sz] - shp) ** 2 / sz
        return ret

    def _compute_approximate_dist_no_beta(self, Xi, Xj):
        sti = self._shapelet_transform(Xi)
        stj = self._shapelet_transform(Xj)
        return self._compute_approximate_dist_from_transforms_no_beta(sti, stj)

    def _compute_approximate_dist_from_transforms_no_beta(self, STi, STj):
        return numpy.linalg.norm(STi - STj, ord=2)

    def _compute_approximate_dist(self, Xi, Xj):
        return self.beta * self._compute_approximate_dist_no_beta(Xi, Xj)

    def _compute_approximate_dist_from_transforms(self, STi, STj):
        return self.beta * self._compute_approximate_dist_from_transforms_no_beta(STi, STj)

    def copy(self):
        return self.derive_model(0, 0, 0, 0)

    def derive_model(self, k, l, d, h):
        m = self.__class__(shapelet_lengths=self.shapelet_lengths, convergence_rate=self.convergence_rate,
                           niter=self.niter, init_method=self.init_method, ada_grad=self.ada_grad,
                           print_loss_every=self.print_loss_every)
        for idx_k in range(self.n_shapelets):
            m.shapelets.append(self.shapelets[idx_k])
            if idx_k == k:
                m.shapelets[idx_k][l, d] += h
        return m

    def get_dist(self, X, i, j):
        if (i, j) not in self.precomputed_dists.keys():
            self.precomputed_dists[i, j] = self.metric(X[i], X[j])
        return self.precomputed_dists[i, j]

    def _precompute_distances(self, X):
        _X = X.reshape((X.shape[0], -1, self.d))
        for i in range(_X.shape[0]):
            for j in range(i + 1, _X.shape[0]):
                self.get_dist(_X, i, j)


class ConvMimicModelL2(MimicModelL2):
    def __init__(self, shapelet_lengths, **kwargs):
        MimicModelL2.__init__(self, shapelet_lengths, **kwargs)

    def _idx_match(self, Xi, k):
        shp = self.shapelets[k]
        if self.d == 1:  # Efficient
            convs = numpy.correlate(Xi.reshape((-1, )), shp.reshape((-1, )), mode="valid")
        else:  # TODO: make it more efficient using correlate?
            sz = shp.shape[0]
            convs = numpy.array([numpy.sum(Xi[t:t+sz] * shp) for t in range(Xi.shape[0] - sz + 1)])
        return numpy.argmax(convs)

    def _shapelet_transform(self, Xi):
        ret = numpy.empty((self.n_shapelets, ))
        for k in range(self.n_shapelets):
            shp = self.shapelets[k]
            sz = shp.shape[0]
            ti = self._idx_match(Xi, k)
            ret[k] = numpy.sum(Xi[ti:ti+sz] * shp) / sz
        return ret

    def _dM_dSkl(self, Xi, ti, Xj, tj, sz):
        return (Xi[ti:ti+sz] - Xj[tj:tj+sz]) / sz


class MimicBetaInitModelL2(MimicModelL2):
    def __init__(self, shapelet_lengths, **kwargs):
        MimicModelL2.__init__(self, shapelet_lengths, **kwargs)

    def _init_shapelets(self, X):
        MimicModelL2._init_shapelets(self, X)
        n_pairs = 100
        pairs = numpy.random.choice(X.shape[0], size=n_pairs * 2).reshape((-1, 2))
        for i in range(n_pairs):
            while pairs[i, 0] == pairs[i, 1]:
                pairs[i, 1] = numpy.random.choice(X.shape[0])
            pairs[i] = min(pairs[i]), max(pairs[i])
            self.get_dist(X, pairs[i, 0], pairs[i, 1])
        self.beta = self._compute_beta(X[pairs[:, 0]], X[pairs[:, 1]], [self.precomputed_dists[i, j] for i, j in pairs])
        print("Initialized beta at value: %f" % self.beta)

    def _compute_beta(self, Xi_s, Xj_s, dists):
        STi_s = [self._shapelet_transform(Xi) for Xi in Xi_s]
        STj_s = [self._shapelet_transform(Xj) for Xj in Xj_s]
        num = 0.
        denom = 0.
        for STi, STj, d_ij in zip(STi_s, STj_s, dists):
            approx_d_ij = self._compute_approximate_dist_from_transforms_no_beta(STi, STj)
            num += approx_d_ij * d_ij
            denom += approx_d_ij ** 2
        denom = max(denom, 1e-6)
        return num / denom


class MimicBetaInitModelConvL2(MimicBetaInitModelL2, ConvMimicModelL2):
    def __init__(self, shapelet_lengths, **kwargs):
        MimicBetaInitModelL2.__init__(self, shapelet_lengths, **kwargs)

    def _init_shapelets(self, X):
        return MimicBetaInitModelL2._init_shapelets(self, X)

    def _compute_beta(self, Xi_s, Xj_s, dists):
        return MimicBetaInitModelL2._compute_beta(self, Xi_s, Xj_s, dists)

    def _idx_match(self, Xi, k):
        return ConvMimicModelL2._idx_match(self, Xi, k)

    def _shapelet_transform(self, Xi):
        return ConvMimicModelL2._shapelet_transform(self, Xi)

    def _dM_dSkl(self, Xi, ti, Xj, tj, sz):
        return ConvMimicModelL2._dM_dSkl(self, Xi, ti, Xj, tj, sz)


class MimicModelIncremental(MimicBetaInitModelL2):
    def __init__(self, shapelet_lengths, size_shapelet_groups=1, **kwargs):
        MimicBetaInitModelL2.__init__(self, shapelet_lengths, **kwargs)
        self.size_shapelet_groups = size_shapelet_groups

    def partial_fit(self, X, niter=None, idx_iter_start=0):
        if niter is None:
            niter = self.niter
        _X = X.reshape((X.shape[0], -1, self.d))
        # TODO
        n_groups = 123
        for group_id in range(n_groups):
            idx_shp_start = group_id * self.size_shapelet_groups
            idx_shp_end = idx_shp_start + self.size_shapelet_groups
            for idx_iter in range(idx_iter_start, idx_iter_start + niter):
                self._update_shapelets_one_iter(_X, idx_iter, idx_start=idx_shp_start, idx_end=idx_shp_end)
                avg_dist = numpy.mean(list(self.precomputed_dists.values()))
                if (idx_iter + 1) % self.print_loss_every == 0:
                    if self.print_approx_loss:
                        loss = self._approximate_loss(_X)
                    else:
                        loss = self._loss(_X)
                    print("iteration %d, loss=%f (average dist: %f, beta=%f)" % (idx_iter + 1, loss, avg_dist, self.beta))

    def _update_shapelets_one_iter(self, X, iter, idx_start, idx_end):
        grad_S, grad_Beta = self._grad(X, idx_start=idx_start, idx_end=idx_end)  # TODO
        if self.ada_grad:
            # TODO
            pass
        else:
            for k in range(self.n_shapelets):
                self.shapelets[k][idx_start:idx_end] -= self.convergence_rate * grad_S[k]
            new_beta = self.beta - self.convergence_rate * grad_Beta
            conv_rate_beta = self.convergence_rate
        if new_beta < 0.:
            grad_Beta = self.beta / conv_rate_beta
            new_beta = 0.
        self.beta = new_beta
        if self.ada_grad:
            # TODO
            for k in range(self.n_shapelets):
                self.past_gradients_Skl[k] += grad_S[k] ** 2
            self.past_gradients_Beta += grad_Beta ** 2

    def _grad(self, X, idx_start, idx_end):
        # TODO
        return None