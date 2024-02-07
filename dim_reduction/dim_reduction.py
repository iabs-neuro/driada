from .dr_base import *
from .data import MVData
from .graph import ProximityGraph
from .embedding import Embedding


# TODO: refactor this
def dr_series(d,
              n_jumps,
              all_metric_params,
              all_graph_params,
              all_embedding_params,
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
    n_iter = maxiter * int(recalculate_if_error) + 1
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
        datalist = [d, MVData(emb.coords, labels=d.labels)]

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

        datalist.append(MVData(emb.coords, labels=d.labels))

    return emb
