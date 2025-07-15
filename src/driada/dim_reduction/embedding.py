
from scipy.sparse.linalg import eigs
import umap.umap_ as umap
from pydiffmap import diffusion_map as dm
from scipy.sparse.csgraph import shortest_path

from sklearn.decomposition import PCA
from sklearn.manifold import spectral_embedding, Isomap, LocallyLinearEmbedding, TSNE
# from sklearn.cluster.spectral import discretize

# warnings.filterwarnings("ignore")

from .dr_base import *
from .graph import ProximityGraph
from .neural import *
from .mvu import *

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
        if kwargs is None:
            kwargs = dict()
        fn = getattr(self, 'create_' + self.e_method_name + '_embedding_')

        if self.e_method.requires_graph:
            # TODO: move connectivity check to graph
            if not self.graph.is_connected():
                raise Exception('Graph is not connected!')

        fn(**kwargs)

    def create_pca_embedding_(self, verbose=True):
        if verbose:
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
        spmatrix = shortest_path(A.todense(), method='D', directed=False)
        self.coords = map.fit_transform(spmatrix).T
        self.reducer_ = map

    def create_mds_embedding_(self):
        """Create MDS (Multi-Dimensional Scaling) embedding."""
        from sklearn.manifold import MDS
        
        # MDS typically uses a distance matrix
        if hasattr(self, 'init_distmat') and self.init_distmat is not None:
            # Use provided distance matrix
            mds = MDS(n_components=self.dim, dissimilarity='precomputed', random_state=42)
            self.coords = mds.fit_transform(self.init_distmat).T
        else:
            # Compute from data
            mds = MDS(n_components=self.dim, random_state=42)
            self.coords = mds.fit_transform(self.init_data.T).T
        
        self.reducer_ = mds


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
        # Convert to numpy array instead of matrix to avoid sklearn compatibility issues
        vecs = spectral_embedding(np.asarray(A.todense()), n_components=dim, eigen_solver=None,
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
                             enc_kwargs=None, dec_kwargs=None,
                             feature_dropout=0.2, train_size=0.8, inter_dim=100,
                             verbose=True,
                             add_corr_loss=False, corr_hyperweight=0,
                             add_mi_loss=False, mi_hyperweight=0, minimize_mi_data=None,
                             log_every=1,
                             device=None):

        # ---------------------------------------------------------------------------

        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # TODO: add train_test_split
        train_dataset = NeuroDataset(self.init_data[:, :int(train_size * self.init_data.shape[1])])
        test_dataset = NeuroDataset(self.init_data[:, int(train_size * self.init_data.shape[1]):])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # ---------------------------------------------------------------------------
        if device is None:
            #  use gpu if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if verbose:
                print('device:', device)

        if not continue_learning:
            # create a model from `AE` autoencoder class
            # load it to the specified device, either gpu or cpu
            model = AE(orig_dim=self.init_data.shape[0], inter_dim=inter_dim, code_dim=self.dim,
                       enc_kwargs=enc_kwargs, dec_kwargs=dec_kwargs, device=device)

            model = model.to(device)
        else:
            model = self.nnmodel

        # create an optimizer object
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # mean-squared error loss
        criterion = nn.MSELoss()

        def correlation_loss(data):
            #print('corr')
            corr = torch.corrcoef(data)
            #print(corr)
            nv = corr.shape[0]
            closs = torch.abs((torch.sum(torch.abs(corr)) - 1*nv)/(nv**2 - nv))  # average pairwise correlation amplitude
            #print(closs)
            return closs

        def data_orthogonality_loss(data, ortdata):

            # punishes big amplitude correlation coefficients between all variables from data and ortdata.
            # ortdata is supposed fixed
            # temporal workaround instead of MINE MI estimation

            # print('ortho')
            #print(ortdata)
            n1, n2 = data.shape[0], ortdata.shape[1]
            fulldata = torch.cat((data, ortdata), dim=0)
            corr = torch.corrcoef(fulldata)
            #print(corr)
            #print(corr[n1:, :n1])
            nvar = n1*n2
            closs = torch.abs(
                (torch.sum(torch.abs(corr))) / nvar)  # average pairwise correlation amplitude
            #print(closs)
            #print()
            return closs

        # ---------------------------------------------------------------------------
        f_dropout = nn.Dropout(feature_dropout)

        best_test_epoch = -1
        best_test_loss = 1e10
        best_test_model = None

        for epoch in range(epochs):
            loss = 0
            for batch_features, _, indices in train_loader:
                batch_features = batch_features.to(device)
                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer.zero_grad()

                # compute reconstructions
                noisy_batch_features = f_dropout(torch.ones(batch_features.shape).to(device)) * batch_features
                outputs = model(noisy_batch_features.float())
                code = model.encoder(noisy_batch_features.float()).T

                '''
                # ==================== MINE experiment ========================

                from torch_mist.estimators import mine
                from torch_mist.utils.train import train_mi_estimator
                from torch_mist.utils import evaluate_mi

                minimize_mi_data = torch.tensor(minimize_mi_data[:, _]).float().to(device)
                print(minimize_mi_data.shape, code.shape)

                estimator = mine(
                    x_dim=minimize_mi_data.shape[0],
                    y_dim=code.shape[0],
                    hidden_dims=[64, 32],
                )


                # Train it on the given samples
                train_log = train_mi_estimator(
                    estimator=estimator,
                    train_data=(minimize_mi_data.T, code.T),
                    batch_size=batch_size,
                    max_iterations=1000,
                    device=device,
                    fast_train=True,
                    max_epochs=10
                )

                # Evaluate the estimator on the entirety of the data
                estimated_mi = evaluate_mi(
                    estimator=estimator,
                    data=(minimize_mi_data.T.to(device), code.T.to(device)),
                    batch_size=batch_size
                )

                print(f"Mutual information estimated value: {estimated_mi} nats")
                # ==================== MINE experiment ========================
                '''
                # compute training reconstruction loss
                train_loss = criterion(outputs, batch_features.float())

                if add_mi_loss:
                    ortdata = torch.tensor(minimize_mi_data[:, indices]).float().to(device)
                    train_loss += mi_hyperweight * data_orthogonality_loss(code, ortdata)

                if add_corr_loss:
                    train_loss += corr_hyperweight * correlation_loss(code)

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()

            # compute the epoch training loss
            loss = loss / len(train_loader)

            # display the epoch training loss
            if (epoch + 1) % log_every == 0:

                # compute loss on test part
                tloss = 0
                for batch_features, _, indices in test_loader:
                    batch_features = batch_features.to(device)
                    # compute reconstructions
                    noisy_batch_features = f_dropout(torch.ones(batch_features.shape).to(device)) * batch_features
                    outputs = model(noisy_batch_features.float())

                    # compute test reconstruction loss
                    test_loss = criterion(outputs, batch_features.float())

                    if add_mi_loss:
                        ortdata = torch.tensor(minimize_mi_data[:, indices]).float().to(device)
                        train_loss += mi_hyperweight * data_orthogonality_loss(code, ortdata)

                    if add_corr_loss:
                        code = model.encoder(noisy_batch_features.float()).T
                        train_loss += corr_hyperweight * correlation_loss(code)

                    tloss += test_loss.item()

                # compute the epoch test loss
                tloss = tloss / len(test_loader)
                if tloss < best_test_loss:
                    best_test_loss = tloss
                    best_test_epoch = epoch + 1
                    best_test_model = model
                if verbose:
                    print(f"epoch : {epoch + 1}/{epochs}, train loss = {loss:.8f}, test loss = {tloss:.8f}")

        if verbose:
            if best_test_epoch != epochs + 1:
                print(f'best model: epoch {best_test_epoch}')

        self.nnmodel = best_test_model
        input_ = torch.tensor(self.init_data.T).float().to(device)
        self.coords = model.get_code_embedding(input_)
        self.nn_loss = tloss

    # -------------------------------------

    def create_vae_embedding_(self, continue_learning=0, epochs=50, lr=1e-3, seed=42, batch_size=32,
                              enc_kwargs=None, dec_kwargs=None, feature_dropout=0.2, kld_weight=1,
                              train_size=0.8, inter_dim=128, verbose=True, log_every=10, **kwargs):

    # TODO: add best model mechanism as above
        # ---------------------------------------------------------------------------
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # TODO: add train_test_split
        # TODO: move out data loading for autoencoders
        train_dataset = NeuroDataset(self.init_data[:, :int(train_size * self.init_data.shape[1])])
        test_dataset = NeuroDataset(self.init_data[:, int(train_size * self.init_data.shape[1]):])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # ---------------------------------------------------------------------------
        #  use gpu if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not continue_learning:
            # create a model from `VAE` autoencoder class
            # load it to the specified device, either gpu or cpu
            model = VAE(orig_dim=len(self.init_data), inter_dim=inter_dim, code_dim=self.dim,
                        enc_kwargs=enc_kwargs, dec_kwargs=dec_kwargs, device=device)
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
            for batch_features, _, _ in train_loader:  # NeuroDataset returns 3 values
                batch_features = batch_features.to(device)
                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer.zero_grad()

                # compute reconstructions
                data = f_dropout(torch.ones(batch_features.shape).to(device)) * batch_features
                data = data.to(device).float()  # Ensure float32
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
            if (epoch + 1) % log_every == 0:
                # compute loss on test part
                tloss = 0
                for batch_features, _, _ in test_loader:  # NeuroDataset returns 3 values
                    data = f_dropout(torch.ones(batch_features.shape).to(device)) * batch_features
                    data = data.to(device).float()  # Ensure float32
                    reconstruction, mu, logvar = model(data)

                    # compute training reconstruction loss
                    mse_loss = criterion(reconstruction, data)
                    kld_loss = -0.5 * torch.sum(
                        1 + logvar - mu.pow(2) - logvar.exp())  # * train_dataset.__len__()/batch_size
                    test_loss = mse_loss + kld_weight * kld_loss
                    tloss += test_loss.item()

                # compute the epoch training loss
                tloss = tloss / len(test_loader)
                if verbose:
                    print(f"epoch : {epoch + 1}/{epochs}, train loss = {loss:.8f}, test loss = {tloss:.8f}")

        self.nnmodel = model
        input_ = torch.tensor(self.init_data.T).float().to(device)
        self.coords = model.get_code_embedding(input_)

        # -------------------------------------

    def continue_learning(self, add_epochs, kwargs={}):
        if self.all_params['e_method_name'] not in ['ae', 'vae']:
            raise Exception('This is not a DL-based method!')

        fn = getattr(self, 'create_' + self.all_params['e_method_name'] + '_embedding_')
        fn(continue_learning=1, epochs=add_epochs, kwargs=kwargs)

