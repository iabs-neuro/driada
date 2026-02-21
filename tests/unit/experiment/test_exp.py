from driada.experiment.exp_base import *
from driada.experiment.synthetic import *
from driada.intense.pipelines import compute_cell_feat_significance


def test_creation(medium_experiment):
    exp = medium_experiment
    assert exp.n_cells == 20  # nneurons
    assert exp.n_frames == 200 * 20  # duration * fps
    assert len(exp.dynamic_features) == 6  # n_dfeats + n_cfeats


def test_intense_exp(medium_experiment):
    exp = medium_experiment
    res = compute_cell_feat_significance(
        exp,
        cell_bunch=None,
        feat_bunch=None,
        data_type="calcium",
        metric="mi",
        mode="two_stage",
        n_shuffles_stage1=100,
        n_shuffles_stage2=5000,

        metric_distr_type="norm",
        noise_ampl=1e-4,
        ds=5,
        topk1=1,
        topk2=5,
        multicomp_correction="holm",
        pval_thr=0.1,
        verbose=True,
        enable_parallelization=False,
    )
    assert res is not None
    assert len(res) > 0
