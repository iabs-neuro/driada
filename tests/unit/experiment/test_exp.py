from driada.experiment.exp_base import *
from driada.experiment.synthetic import *
from driada.intense.pipelines import compute_cell_feat_significance


def test_creation(medium_experiment):
    exp = medium_experiment


def test_intense_exp(medium_experiment):
    exp = medium_experiment
    res_ = compute_cell_feat_significance(
        exp,
        cell_bunch=None,
        feat_bunch=None,
        data_type="calcium",
        metric="mi",
        mode="two_stage",
        n_shuffles_stage1=100,
        n_shuffles_stage2=1000,
        joint_distr=False,
        metric_distr_type="norm",
        noise_ampl=1e-4,
        ds=1,
        topk1=1,
        topk2=5,
        multicomp_correction="holm",
        pval_thr=0.1,
        verbose=True,
        enable_parallelization=False,
    )
