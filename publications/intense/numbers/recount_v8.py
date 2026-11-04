"""Results numbers of the INTENSE paper from the seeded DRIADA 1.2.2 NOF recompute.

Reads <db_dir>/NOF_pre_disent.csv and NOF_post_disent.csv (compute/build_db.py) and prints
every count used in Results; writes <numbers_dir>/recount_v8.json.
Selective pair (paper criterion): p < 0.001, excess MI > 0.04 bits, activity ratio > 1
for binary features (NaN passes); aggregate features 'any place'/'any object' excluded.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402


import pandas as pd

HERE = P.NUMBERS_DIR   # JSON outputs
N_NEURON_SESSIONS = 30105
KEY = ["mouse", "session", "neuron_idx"]
LOCO_OVER_SPEED = ["locomotion", "walk", "rest", "freezing"]


def load(name):
    d = pd.read_csv(P.DB_DIR / f"{name}.csv")
    return d[~d.feature.isin(["any place", "any object"])]


def tool_sig(d):
    """Pairs the tool passes to disentanglement: p < 0.001, anti-selective removed."""
    return d[(d.pval < 0.001) & (d.signal_ratio.isna() | (d.signal_ratio > 1))]


def paper(d):
    d = tool_sig(d)
    return d[d.me_excess > 0.04]


def counts(x):
    g = x.groupby(KEY).feature.nunique()
    return dict(pairs=len(x), selective=len(g), multi=int((g >= 2).sum()),
                n1=int((g == 1).sum()), n2=int((g == 2).sum()), n3plus=int((g >= 3).sum()),
                delay_sign_pct={int(k): round(v, 2) for k, v in
                                x.delay_sign.value_counts(normalize=True).mul(100).sort_index().items()},
                delay_sign_n={int(k): int(v) for k, v in x.delay_sign.value_counts().sort_index().items()},
                per_feature_neurons=x.groupby("feature")[KEY].apply(lambda t: len(t.drop_duplicates())).to_dict())


pre_all, post_all = load("NOF_pre_disent"), load("NOF_post_disent")
pre, post = paper(pre_all), paper(post_all)
out = {"pre": counts(pre), "post": counts(post)}
for k in ("pre", "post"):
    c = out[k]
    c["selective_pct_of_all"] = round(100 * c["selective"] / N_NEURON_SESSIONS, 2)
    c["multi_pct_of_selective"] = round(100 * c["multi"] / c["selective"], 2)
out["multi_reduction_pct"] = round(100 * (1 - out["post"]["multi"] / out["pre"]["multi"]), 2)
out["post_multi_pct_of_pre_selective"] = round(100 * out["post"]["multi"] / out["pre"]["selective"], 2)

# Speed: which speed neurons lost speed, and was a locomotion-state label significant for
# them at tool level (the priority rule applies whenever both are significant)?
tool_pre = tool_sig(pre_all)
spd_pre = set(map(tuple, pre[pre.feature == "speed"][KEY].values))
spd_post = set(map(tuple, post[post.feature == "speed"][KEY].values))
loco_tool = set(map(tuple, tool_pre[tool_pre.feature.isin(LOCO_OVER_SPEED)][KEY].values))
lost = spd_pre - spd_post
out["speed"] = dict(pre=len(spd_pre), post=len(spd_post), lost=len(lost),
                    lost_with_loco_label=len(lost & loco_tool),
                    lost_without_loco_label=len(lost - loco_tool),
                    pre_with_loco_label=len(spd_pre & loco_tool),
                    post_not_in_pre=len(spd_post - spd_pre))
# which of locomotion/walk/rest/freezing
for f in LOCO_OVER_SPEED:
    s = set(map(tuple, tool_pre[tool_pre.feature == f][KEY].values))
    out["speed"][f"lost_with_{f}"] = len(lost & s)
# Body vs head direction
bd_tool = set(map(tuple, tool_pre[tool_pre.feature == "bodydirection_2d"][KEY].values))
hd_pre = set(map(tuple, pre[pre.feature == "headdirection_2d"][KEY].values))
hd_post = set(map(tuple, post[post.feature == "headdirection_2d"][KEY].values))
bd_pre = set(map(tuple, pre[pre.feature == "bodydirection_2d"][KEY].values))
out["direction"] = dict(hd_pre=len(hd_pre), hd_post=len(hd_post), bd_pre=len(bd_pre),
                        bd_post=len(set(map(tuple, post[post.feature == "bodydirection_2d"][KEY].values))),
                        hd_lost_with_bd_label=len((hd_pre - hd_post) & bd_tool),
                        hd_and_bd_pre_paper=len(hd_pre & bd_pre))
# Variants for the record (not used in text): no MI threshold, raw MI threshold
out["variant_no_mi_threshold"] = {k: counts(tool_sig(d))["selective"] for k, d in
                                  (("pre", pre_all), ("post", post_all))}
out["variant_no_mi_threshold_pairs"] = {k: len(tool_sig(d)) for k, d in (("pre", pre_all), ("post", post_all))}
HERE.mkdir(parents=True, exist_ok=True)
json.dump(out, open(HERE / "recount_v8.json", "w"), indent=1, default=int)
print(json.dumps({k: v for k, v in out.items()}, indent=1, default=int))
