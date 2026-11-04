"""Data of Fig. 2a: the example neuron read from the synchronized file, wavelet events of DRIADA 1.2.2
at the frame rate of the session.

For neuron 60 of the earlier export of H31 1D (= row 59 of the synchronized file, events_wvt.py): the scaled
calcium smoothed with a Gaussian of 8 frames ('sig'), its continuous wavelet transform at the frame rate of
the session, and 'sp', the binary event train of the wavelet detector at that rate (events_wvt.py).

usage: python -X utf8 -B figure_chain/figures_v10.3/work/fig2a_data_v103.py
       -> figures_v10.3/work/fig2_out/fig2a_data_v103.npz, figures_v10.3/work/verify/fig2a_data.json
          under [figures] output_dir/figure_chain; needs figure_chain/figures_v10.4/work/events_wvt.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # figure_chain/
import _paths_fig as _F  # noqa: E402  (puts the paper folder on sys.path)

_F.P.use_driada(tools=False)
import scipy.integrate  # noqa: E402

if not hasattr(scipy.integrate, "trapz"):
    scipy.integrate.trapz = scipy.integrate.trapezoid
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from ssqueezepy import cwt  # noqa: E402
from ssqueezepy.wavelets import Wavelet  # noqa: E402
from driada.experiment import load_exp_from_aligned_data  # noqa: E402

V2 = _F.P.NOF_ALIGNED / "NOF_H31_1D_aligned.npz"
EVENTS = _F.data_dir(_F.CHAIN / "figures_v10.4" / "work") / "events" / "events_NOF_H31_1D.npz"
OUT = _F.out_dir(HERE / "fig2_out")
VERIFY = _F.out_dir(HERE / "verify")
SIGMA = 8  # frames, as in compute_fig2a_data and in the Methods


def main():
    ev = np.load(EVENTS)
    row = int(ev["rows_aligned"][0])
    assert int(ev["rows_older_export"][0]) == 60 and row == 59, (ev["rows_older_export"], row)
    fps = float(ev["fps"])
    data = dict(np.load(V2, allow_pickle=True))
    meta = data.pop("_metadata").item()
    data.pop("_sync_info", None)
    assert abs(float(meta["fps"]) - fps) < 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # as data_loader.load_nof_experiment builds it, with the rate of the session
        exp = load_exp_from_aligned_data(
            data_source="NOF", exp_params={"track": "NOF", "animal_id": "H31", "session": "1D"},
            data=data, force_continuous=["x", "y", "speed", "bodydirection", "headdirection"],
            static_features={"fps": fps, "t_rise_sec": 0.05, "t_off_sec": 1.5},
            verbose=False, reconstruct_spikes=False)
    assert abs(float(exp.fps) - fps) < 1e-9
    trace = np.asarray(exp.neurons[row].ca.scdata, float)
    sig = gaussian_filter1d(trace, sigma=SIGMA)
    wavelet = Wavelet(("gmw", {"gamma": 3, "beta": 2, "centered_scale": True}), N=8196)
    W, scales = cwt(sig, wavelet=wavelet, fs=fps)
    Wx = np.abs(W).astype(np.float32)
    sp = ev[f"n{row}_true_sp"].astype(float)
    assert len(sp) == len(sig)
    np.savez_compressed(OUT / "fig2a_data_v103.npz", sig=sig, sp=sp, Wx=Wx, wvt_scales=scales, fps=fps, row=row)
    rep = dict(fps=fps, row_aligned=row, n_frames=len(sig), Wx_max=float(Wx.max()), new_events_frames=int(sp.sum()),
               new_events_periods=int((np.diff(np.r_[0, sp]) == 1).sum()))
    (VERIFY / "fig2a_data.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
