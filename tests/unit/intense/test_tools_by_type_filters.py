"""Tests for running the analysis-tool filters with type-based representations."""

from types import SimpleNamespace

import pytest

from tools.selectivity_dynamics.analysis import run_intense_analysis
from tools.selectivity_dynamics.filters import (
    GENERAL_PRIORITY_RULES,
    build_mi_ratio_filter,
    build_priority_filter,
    tdm_post_filter,
    with_source_feature_names,
)

SOURCES = {
    "place_quad": "place",
    "3d-place_quad": "3d-place",
    "speed_quad": "speed",
    "headdirection_harm2": "headdirection",
    "bodydirection_harm2": "bodydirection",
}


@pytest.fixture
def exp():
    return SimpleNamespace(_representation_sources=dict(SOURCES))


def test_priority_rules_apply_to_derived_names(exp):
    wrapped = with_source_feature_names(
        build_priority_filter(GENERAL_PRIORITY_RULES), exp
    )
    sels = {
        0: ["bodydirection_harm2", "headdirection_harm2"],
        1: ["locomotion", "speed_quad"],
    }
    decisions = {0: {}, 1: {}}
    wrapped(sels, decisions, {0: {}, 1: {}})
    assert decisions[0] == {("bodydirection_harm2", "headdirection_harm2"): 0}
    assert decisions[1] == {("locomotion", "speed_quad"): 0}
    assert sels == {
        0: ["bodydirection_harm2", "headdirection_harm2"],
        1: ["locomotion", "speed_quad"],
    }


def test_mi_ratio_compares_quadratic_place_representations(exp):
    wrapped = with_source_feature_names(
        build_mi_ratio_filter(("place", "3d-place")), exp
    )
    sels = {0: ["place_quad", "3d-place_quad"]}
    decisions = {0: {}}
    stats = {0: {"place_quad": {"me": 0.30}, "3d-place_quad": {"me": 0.10}}}
    wrapped(
        neuron_selectivities=sels,
        pair_decisions=decisions,
        renames={0: {}},
        cell_feat_stats=stats,
        feat_names=["place_quad", "3d-place_quad"],
    )
    assert decisions == {0: {("place_quad", "3d-place_quad"): 0}}


def test_post_filter_tie_break_on_derived_names(exp):
    wrapped = with_source_feature_names(tdm_post_filter, exp)
    per_neuron = {
        5: {
            "pairs": {("place_quad", "3d-place_quad"): {"result": 0.5}},
            "renames": {},
            "final_sels": ["place_quad", "3d-place_quad"],
        }
    }
    wrapped(per_neuron_disent=per_neuron, cell_feat_stats={}, feat_names=[])
    info = per_neuron[5]["pairs"][("place_quad", "3d-place_quad")]
    assert info["result"] == 0
    assert info["source"] == "post_filter_tiebreak"


def test_without_derived_features_filter_is_unchanged():
    raw_exp = SimpleNamespace()
    wrapped = with_source_feature_names(build_priority_filter([("a", "b")]), raw_exp)
    decisions = {0: {}}
    wrapped({0: ["a", "b"]}, decisions, {0: {}})
    assert decisions == {0: {("a", "b"): 0}}


def test_by_type_requires_mi_metric():
    config = {"metric": "fast_pearsonr", "representation": "by_type"}
    with pytest.raises(ValueError, match="by_type"):
        run_intense_analysis(SimpleNamespace(dynamic_features={}), config, [])


class _Captured(Exception):
    """Stops run_intense_analysis once the pipeline arguments are known."""


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"metric": "mi"}, "by_type"),
        ({"metric": "fast_pearsonr"}, "raw"),
        ({"metric": "mi", "representation": "raw"}, "raw"),
    ],
    ids=["mi-default", "non-mi-default", "explicit-raw"],
)
def test_representation_passed_to_pipeline(monkeypatch, config, expected):
    import driada

    def fake_pipeline(exp, **kwargs):
        raise _Captured(kwargs["representation"])

    monkeypatch.setattr(driada, "compute_cell_feat_significance", fake_pipeline)
    config = {
        "n_shuffles_stage1": 10,
        "n_shuffles_stage2": 100,
        "ds": 1,
        "pval_thr": 0.01,
        "multicomp_correction": None,
        "engine": "auto",
        **config,
    }
    with pytest.raises(_Captured, match=f"^{expected}$"):
        run_intense_analysis(SimpleNamespace(dynamic_features={}), config, [])


def test_cross_analysis_loader_maps_derived_names_to_plain_ones():
    import pandas as pd

    from tools.neuron_database.database import pretransform_representation_names

    data = pd.DataFrame({"feature": [
        "place_quad", "speed_quad", "headdirection_harm2", "bodydirection_harm2",
        "place", "headdirection_2d", "walls", "place-corners",
    ]})
    out = pretransform_representation_names(data)
    assert list(out["feature"]) == [
        "place", "speed", "headdirection_2d", "bodydirection_2d",
        "place", "headdirection_2d", "walls", "place-corners",
    ]
    assert list(data["feature"])[0] == "place_quad"

