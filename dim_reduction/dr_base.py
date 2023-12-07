# @title Method class { form-width: "300px" }
from pynndescent.distances import named_distances
import sys
from .data import Data

class DRMethod(object):

    def __init__(self, is_linear, requires_graph, requires_distmat, nn_based):
        self.is_linear = is_linear
        self.requires_graph = requires_graph
        self.requires_distmat = requires_distmat
        self.nn_based = nn_based


METHODS_DICT = {
    'pca': DRMethod(1, 0, 0, 0),
    'le': DRMethod(0, 1, 0, 0),
    'auto_le': DRMethod(0, 1, 0, 0),
    'dmaps': DRMethod(0, 1, 0, 0),
    'auto_dmaps': DRMethod(0, 1, 0, 0),
    'mds': DRMethod(0, 0, 1, 0),
    'isomap': DRMethod(0, 1, 0, 0),
    'lle': DRMethod(0, 1, 0, 0),
    'hlle': DRMethod(0, 1, 0, 0),
    'mvu': DRMethod(0, 1, 0, 0),
    'ae': DRMethod(0, 0, 0, 1),
    'vae': DRMethod(0, 0, 0, 1),
    'tsne': DRMethod(0, 0, 0, 0),
    'umap': DRMethod(0, 1, 0, 0)
}

GRAPH_CONSTRUCTION_METHODS = ['knn', 'auto_knn', 'eps', 'eknn', 'umap', 'tsne']

EMBEDDING_CONSTRUCTION_METHODS = ['pca',
                                  'le',
                                  'auto_le',
                                  'dmaps',
                                  'auto_dmaps',
                                  'mds',
                                  'isomap',
                                  'lle',
                                  'hlle',
                                  'mvu',
                                  'ae',
                                  'vae',
                                  'tsne',
                                  'umap']


def m_param_filter(para):
    '''
    This function prunes parameters that are excessive for
    chosen distance matrix construction method
    '''
    name = para['metric_name']
    appr_keys = ['metric_name']

    if not (para['sigma'] is None):
        appr_keys.append('sigma')

    if name not in named_distances:
        if name == 'hyperbolic':
            # para['metric_name'] = globals()[name]
            pass
        else:
            raise Exception('this custom metric is not implemented!')

    if name == 'minkowski':
        appr_keys.append('p')

    return {key: para[key] for key in appr_keys}


def g_param_filter(para):
    '''
    This function prunes parameters that are excessive for
    chosen graph construction method
    '''
    gmethod = para['g_method_name']
    appr_keys = ['g_method_name', 'max_deleted_nodes', 'weighted', 'dist_to_aff']

    if gmethod in ['knn', 'auto_knn', 'umap']:
        appr_keys.extend(['nn'])

    elif gmethod == 'eps':
        appr_keys.extend(['eps', 'eps_min'])

    elif gmethod == 'eknn':
        appr_keys.extend(['eps', 'eps_min', 'nn'])

    elif gmethod == 'tsne':
        appr_keys.extend(['perplexity'])

    return {key: para[key] for key in appr_keys}


def e_param_filter(para):
    '''
    This function prunes parameters that are excessive for the
    chosen embedding construction method
    '''

    appr_keys = ['e_method', 'e_method_name', 'dim']

    if para['e_method'].requires_graph:
        appr_keys.append('check_graph_connectivity')

    if para['e_method_name'] == 'umap':
        appr_keys.append('min_dist')

    if para['e_method_name'] in ['dmaps', 'auto_dmaps']:
        appr_keys.append('dm_alpha')

    return {key: para[key] for key in appr_keys}


def jump_series(d, n_jumps, all_metric_params, all_graph_params, all_embedding_params,
                recalculate_if_error=0):
    print('---------------------------  JUMP 1  --------------------------------')
    print('Performing jump from dim', d.dim, 'to dim', all_embedding_params['dim'][0], ':')
    metric_params = dict(zip(all_metric_params.keys(), [val[0] for val in all_metric_params.values()]))
    graph_params = dict(zip(all_graph_params.keys(), [val[0] for val in all_graph_params.values()]))
    embedding_params = dict(zip(all_embedding_params.keys(), [val[0] for val in all_embedding_params.values()]))
    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]

    m_params = m_param_filter(metric_params)
    g_params = g_param_filter(graph_params)
    e_params = e_param_filter(embedding_params)

    maxiter = 20
    n_iter = maxiter * (recalculate_if_error) + 1
    it = 0
    success = 0
    while it < n_iter and success == 0:
        try:
            emb = d.get_embedding(m_params, g_params, e_params)
            success = 1
        except:
            it += 1
            print("Unexpected error:", sys.exc_info()[0])
            raise

    if it == n_iter:
        raise Exception('First jump failed after %s attempts' % n_iter)

    if n_jumps > 1:
        datalist = [d, Data(emb.coords, labels=d.labels)]

    for i in range(1, n_jumps):
        print('---------------------------  JUMP ' + str(i + 1) + ' --------------------------------', )
        print('Performing jump from dim', datalist[i].dim, 'to dim', all_embedding_params['dim'][i], ':')
        metric_params = dict(zip(all_metric_params.keys(), [val[i] for val in all_metric_params.values()]))
        graph_params = dict(zip(all_graph_params.keys(), [val[i] for val in all_graph_params.values()]))
        embedding_params = dict(zip(all_embedding_params.keys(), [val[i] for val in all_embedding_params.values()]))
        embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]

        m_params = m_param_filter(metric_params)
        g_params = g_param_filter(graph_params)
        e_params = e_param_filter(embedding_params)

        maxiter = 20
        n_iter = maxiter * (recalculate_if_error) + 1
        it = 0
        success = 0

        while it < n_iter and success == 0:
            try:
                emb = datalist[i].get_embedding(m_params, g_params, e_params)
                success = 1
            except:
                print('iter', it)
                it += 1

        if it == n_iter:
            raise Exception('Jump ', str(i + 1), ' failed after %s attempts' % n_iter)

        datalist.append(Data(emb.coords, labels=d.labels))

    return emb
