
from scipy.sparse.linalg import eigs
import scipy
import numpy as np
from itertools import combinations
import umap.umap_ as umap
import scipy.sparse as sp
from pydiffmap import diffusion_map as dm
from scipy.sparse.csgraph import shortest_path

from sklearn.decomposition import PCA
from sklearn.manifold import spectral_embedding, Isomap, LocallyLinearEmbedding, TSNE
from sklearn.cluster import KMeans, SpectralClustering
# from sklearn.cluster.spectral import discretize
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, homogeneity_score, completeness_score, \
    v_measure_score

import warnings
# warnings.filterwarnings("ignore")

from .dr_base import *
from .graph import ProximityGraph
from .neural import *
#from .mvu import *

from ..network.matrix_utils import get_inv_sqrt_diag_matrix

def norm_cross_corr(a, b):
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    c = np.correlate(a, b, 'full')
    return c


def remove_outliers(data, thr_percentile):
    thr1 = np.mean(data) + thr_percentile * np.std(data)
    thr2 = np.mean(data) - thr_percentile * np.std(data)
    good_points = np.where((data < thr1) & (data > thr2))[0]

    return good_points, data[good_points]


class Embedding:
    '''
    Low-dimensional representation of data
    '''

    def __init__(self, init_data, init_distmat, labels, params, g=None):
        if g is not None:
            if isinstance(g, ProximityGraph):
                self.graph = g
            else:
                raise Exception('Wrong graph type!')

        self.all_params = e_param_filter(params)
        for key in params:
            setattr(self, key, params[key])

        self.init_data = init_data
        self.init_distmat = init_distmat

        if self.e_method.is_linear:
            self.transformation_matrix = None

        self.labels = labels
        self.coords = None

        try:
            self.nclasses = len(set(self.labels))
        except:
            self.nclasses = np.unique(self.labels)

        if self.e_method.nn_based:
            self.nnmodel = None

    def build(self, kwargs=None):
        fn = getattr(self, 'create_' + self.e_method_name + '_embedding_')

        if self.e_method.requires_graph:
            # TODO: move connectivity check to graph
            if not self.graph.is_connected():
                raise Exception('Graph is not connected!')

        fn(**kwargs)

    def create_pca_embedding_(self):
        print('Calculating PCA embedding...')
        pca = PCA(n_components=self.dim)
        self.coords = pca.fit_transform(self.init_data.T).T
        self.reducer_ = pca
        # print(pca.explained_variance_ratio_)

    def create_isomap_embedding_(self):
        A = self.graph.adj
        map = Isomap(n_components=self.dim, n_neighbors=self.graph.nn,
                     metric='precomputed')
        # self.coords = sp.csr_matrix(map.fit_transform(self.graph.data.A.T).T)
        spmatrix = shortest_path(A.A, method='D', directed=False)
        self.coords = map.fit_transform(spmatrix).T
        self.reducer_ = map


    def create_mvu_embedding_(self):
        mvu = MaximumVarianceUnfolding(equation="berkley", solver=cp.SCS, solver_tol=1e-2,
                                       eig_tol=1.0e-10, solver_iters=2500,
                                       warm_start=False, seed=None)

        self.coords = mvu.fit_transform(self.graph.data.T, self.dim,
                                                      self.graph.nn, dropout_rate=0)
        self.reducer_ = mvu

    def create_lle_embedding_(self):
        lle = LocallyLinearEmbedding(n_components=self.dim, n_neighbors=self.graph.nn)
        self.coords = lle.fit_transform(self.graph.data.T).T
        self.reducer_ = lle

    def create_hlle_embedding_(self):
        hlle = LocallyLinearEmbedding(n_components=self.dim,
                                      n_neighbors=self.graph.nn,
                                      method='hessian')
        self.coords = hlle.fit_transform(self.graph.data.T).T
        self.reducer_ = hlle

    def create_le_embedding_(self):
        A = self.graph.adj
        dim = self.dim
        n = self.graph.n

        DH = get_inv_sqrt_diag_matrix(A)
        P = self.graph.get_matrix('trans')

        start_v = np.ones(n)
        # LR mode is much more stable, this is why we use P matrix largest eigenvalues
        eigvals, eigvecs = eigs(P, k=dim + 1, which='LR', v0=start_v, maxiter=n * 1000)
        # eigvals, vecs = eigs(nL, k = dim2 + 1, which = 'SM')

        eigvals = np.asarray([np.round(np.real(x), 6) for x in eigvals])

        if np.count_nonzero(eigvals == 1.0) > 1:
            raise Exception('Graph is not connected, LE will result in errors!')
        else:
            vecs = np.real(eigvecs.T[1:])
            vec_norms = np.array([np.real(sum([x * x for x in v])) for v in vecs])
            vecs = vecs / vec_norms[:, np.newaxis]
            vecs = vecs.dot(DH.toarray())
            self.coords = vecs

    def create_auto_le_embedding_(self):
        A = self.graph.adj
        dim = self.dim

        A = A.asfptype()
        vecs = spectral_embedding(A.A, n_components=dim, eigen_solver=None,
                                  random_state=None, eigen_tol=0.0, norm_laplacian=True, drop_first=True).T

        self.coords = vecs

    def create_dmaps_embedding_(self):
        raise Exception('not so easy to implement properly (https://sci-hub.se/10.1016/j.acha.2015.01.001)')

    def create_auto_dmaps_embedding_(self):
        dim = self.dim
        nn = self.graph.nn
        metric = self.graph.metric
        metric_args = self.graph.metric_args
        alpha = self.dm_alpha if hasattr(self, 'dm_alpha') else 1

        mydmap = dm.DiffusionMap.from_sklearn(n_evecs=dim,
                                              k=nn,
                                              epsilon='bgh',
                                              metric=metric,
                                              metric_params=metric_args,
                                              alpha=alpha)

        dmap = mydmap.fit_transform(self.init_data.T)

        self.coords = dmap.T
        self.reducer_ = dmap

    def create_tsne_embedding_(self):
        model = TSNE(n_components=self.dim, verbose=1)
        self.coords = model.fit_transform(self.init_data.T).T
        self.reducer_ = model

    def create_umap_embedding_(self):
        min_dist = self.min_dist
        reducer = umap.UMAP(n_neighbors=self.graph.nn, n_components=self.dim,
                            min_dist=min_dist)

        self.coords = reducer.fit_transform(self.graph.data.T).T
        self.reducer_ = reducer

    def create_ae_embedding_(self, continue_learning=0, epochs=50, lr=1e-3, seed=42, batch_size=32,
                             enc_kwargs=None, dec_kwargs=None, feature_dropout=0.2, train_size=0.8):

        # ---------------------------------------------------------------------------

        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # TODO: add train_test_split
        train_dataset = NeuroDataset(self.init_data[:, :int(0.8*self.init_data.shape[1])])
        test_dataset = NeuroDataset(self.init_data[:, int(0.8 * self.init_data.shape[1]):])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # ---------------------------------------------------------------------------
        #  use gpu if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not continue_learning:
            # create a model from `AE` autoencoder class
            # load it to the specified device, either gpu or cpu
            model = AE(orig_dim=self.init_data.shape[0], inter_dim=100, code_dim=self.dim,
                       enc_kwargs=enc_kwargs, dec_kwargs=dec_kwargs, device=device)

            model = model.to(device)
        else:
            model = self.nnmodel

        # create an optimizer object
        # Adam optimizer with learning rate 1e-3
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # mean-squared error loss
        criterion = nn.MSELoss()

        # ---------------------------------------------------------------------------
        f_dropout = nn.Dropout(feature_dropout)

        for epoch in range(epochs):
            loss = 0
            for batch_features, _ in train_loader:
                batch_features = batch_features.to(device)
                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer.zero_grad()

                # compute reconstructions
                noisy_batch_features = f_dropout(torch.ones(batch_features.shape).to(device)) * batch_features
                outputs = model(noisy_batch_features.float())

                # compute training reconstruction loss
                train_loss = criterion(outputs, batch_features.float())

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()

            # compute the epoch training loss
            loss = loss / len(train_loader)

            # display the epoch training loss
            if (epoch + 1) % 10 == 0:

                # compute loss on test part
                tloss = 0
                for batch_features, _ in test_loader:
                    batch_features = batch_features.to(device)
                    # compute reconstructions
                    noisy_batch_features = f_dropout(torch.ones(batch_features.shape).to(device)) * batch_features
                    outputs = model(noisy_batch_features.float())

                    # compute training reconstruction loss
                    test_loss = criterion(outputs, batch_features.float())
                    tloss += test_loss.item()

                # compute the epoch training loss
                tloss = tloss / len(test_loader)
                print(f"epoch : {epoch + 1}/{epochs}, train loss = {loss:.8f}, test loss = {tloss:.8f}")

        self.nnmodel = model
        input_ = torch.tensor(self.init_data.T).float().to(device)
        self.coords = model.get_code_embedding(input_)

    # -------------------------------------

    def create_vae_embedding_(self, continue_learning=0, epochs=50, lr=1e-3, seed=42, batch_size=32,
                              enc_kwargs=None, dec_kwargs=None, feature_dropout=0.2, kld_weight=1):

        # ---------------------------------------------------------------------------
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # TODO: add train_test_split
        train_dataset = NeuroDataset(self.init_data[:, :int(0.8*self.init_data.shape[1])])
        test_dataset = NeuroDataset(self.init_data[:, int(0.8 * self.init_data.shape[1]):])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # ---------------------------------------------------------------------------
        #  use gpu if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not continue_learning:
            # create a model from `VAE` autoencoder class
            # load it to the specified device, either gpu or cpu
            model = VAE(orig_dim=len(self.graph.data.A), inter_dim=128, code_dim=self.dim,
                        enc_kwargs=enc_kwargs, dec_kwargs=dec_kwargs)
            model = model.to(device)
        else:
            model = self.nnmodel

        # create an optimizer object
        # Adam optimizer with learning rate lr
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # BCE error loss
        #criterion = nn.BCELoss(reduction='sum')
        criterion = nn.MSELoss()

        # ---------------------------------------------------------------------------
        f_dropout = nn.Dropout(feature_dropout)

        for epoch in range(epochs):
            loss = 0
            loss1 = 0
            loss2 = 0
            for batch_features, _ in train_loader:
                batch_features = batch_features.to(device)
                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer.zero_grad()

                # compute reconstructions
                data = f_dropout(torch.ones(batch_features.shape).to(device)) * batch_features
                data = data.to(device)
                reconstruction, mu, logvar = model(data)

                # compute training reconstruction loss
                mse_loss = criterion(reconstruction, data)
                kld_loss = -0.5 * torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp())  # * train_dataset.__len__()/batch_size
                train_loss = mse_loss + kld_weight*kld_loss

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()
                loss1 += mse_loss.item()
                loss2 += kld_loss.item()

            # compute the epoch training loss
            loss = loss / len(train_loader)
            loss1 = loss1 / len(train_loader)
            loss2 = loss2 / len(train_loader)

            # display the epoch training loss
            # display the epoch training loss
            if (epoch + 1) % 10 == 0:
                # compute loss on test part
                tloss = 0
                for batch_features, _ in test_loader:
                    data = f_dropout(torch.ones(batch_features.shape).to(device)) * batch_features
                    data = data.to(device)
                    reconstruction, mu, logvar = model(data)

                    # compute training reconstruction loss
                    mse_loss = criterion(reconstruction, data)
                    kld_loss = -0.5 * torch.sum(
                        1 + logvar - mu.pow(2) - logvar.exp())  # * train_dataset.__len__()/batch_size
                    test_loss = mse_loss + kld_weight * kld_loss
                    tloss += test_loss.item()

                # compute the epoch training loss
                tloss = tloss / len(test_loader)
                print(f"epoch : {epoch + 1}/{epochs}, train loss = {loss:.8f}, test loss = {tloss:.8f}")

        self.nnmodel = model
        input_ = torch.tensor(self.init_data.T).float().to(device)
        self.coords = model.get_code_embedding(input_)

        # -------------------------------------

    def continue_learning(self, add_epochs):
        if self.all_params['e_method_name'] not in ['ae', 'vae']:
            raise Exception('This is not a DL-based method!')

        fn = getattr(self, 'create_' + self.all_params['e_method_name'] + '_embedding_')
        fn(continue_learning=1, epochs=add_epochs)

