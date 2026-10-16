"""DCI robustness sweep for Fig. 5 and the supplementary figure on initializations.

Same CANN, experiment and autoencoder settings as the Fig. 5 data script
(fig5/fig5_data.py); only the AE initialization seed varies. Panel B of Fig. 5
shows the best run of each loss, read from the cache file written here. Two
losses: reconstruction only, and reconstruction + correlation (weight 1.0),
12 seeds each.

Results are saved after every run, so a restarted job skips finished runs.
"""
import ctypes
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import elife_paths
FIG5_DIR = os.path.join(HERE, '..', 'fig5')
sys.path.insert(0, FIG5_DIR)
OUT = os.path.join(elife_paths.CACHE_DIR, 'fig5_ae_sweep.npz')

CORR_WEIGHTS = [0.0, 1.0]
AE_SEEDS = [42, 7777] + list(range(100, 110))


def high_qos():
    """Opt this process out of Windows power throttling: background jobs are
    otherwise given a small fraction of a core."""
    if os.name != 'nt':
        return

    class PPT(ctypes.Structure):
        _fields_ = [('Version', ctypes.c_ulong), ('ControlMask', ctypes.c_ulong),
                    ('StateMask', ctypes.c_ulong)]
    k = ctypes.WinDLL('kernel32')
    st = PPT(1, 1, 0)
    ok = k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st))
    print(f"[{'OK' if ok else 'FAILED'}] high QoS for this process", flush=True)


def main():
    high_qos()
    from fig5_data import (generate_cann_inputs, simulate_cann, build_experiment,
                           _compute_separation_metrics, SEED, FPS, DURATION)

    rng = np.random.default_rng(SEED)
    inputs = generate_cann_inputs(int(DURATION * FPS), FPS, rng)
    activations, phi1, phi2 = simulate_cann(inputs, rng)
    exp = build_experiment(activations, inputs)
    print(f'{exp.n_cells} neurons, {exp.n_frames} frames', flush=True)

    shape = (len(AE_SEEDS), len(CORR_WEIGHTS))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        prev = np.load(OUT)
        dci, r1, r2 = prev['dci_matrix'], prev['r2_t1_matrix'], prev['r2_t2_matrix']
        coords_all = prev['coords']
        dci_pca = float(prev['dci_pca'])
    else:
        dci, r1, r2 = (np.full(shape, np.nan) for _ in range(3))
        coords_all = np.full(shape + (3, exp.n_frames), np.nan, dtype=np.float32)
        pca = exp.create_embedding('pca', n_components=3).T
        dci_pca = _compute_separation_metrics(pca, inputs)['disentanglement']
    print(f'PCA D = {dci_pca:.3f}', flush=True)

    for wi, cw in enumerate(CORR_WEIGHTS):
        for si, s in enumerate(AE_SEEDS):
            if not np.isnan(dci[si, wi]):
                continue
            t0 = time.time()
            loss = [{'name': 'reconstruction', 'weight': 1.0, 'loss_type': 'mse'}]
            if cw > 0:
                loss.append({'name': 'correlation', 'weight': cw})
            emb = exp.create_embedding('flexible_ae', n_components=3, architecture='ae',
                                       inter_dim=128, loss_components=loss, epochs=200,
                                       lr=1e-3, batch_size=512, verbose=False, seed=s)
            coords = emb.T
            m = _compute_separation_metrics(coords, inputs)
            dci[si, wi], r1[si, wi], r2[si, wi] = (m['disentanglement'], m['r2_theta1'],
                                                   m['r2_theta2'])
            coords_all[si, wi] = coords
            np.savez(OUT, dci_pca=dci_pca, dci_matrix=dci, r2_t1_matrix=r1, r2_t2_matrix=r2,
                     corr_weights=np.array(CORR_WEIGHTS), ae_seeds=np.array(AE_SEEDS),
                     coords=coords_all)
            print(f'cw={cw} seed={s}: D={dci[si, wi]:.3f} R2(t1)={r1[si, wi]:.3f} '
                  f'R2(t2)={r2[si, wi]:.3f} ({time.time() - t0:.0f}s)', flush=True)

    print('\nSUMMARY', flush=True)
    print(f'PCA D = {dci_pca:.3f}')
    for wi, cw in enumerate(CORR_WEIGHTS):
        c = dci[:, wi]
        b = int(np.argmax(c))
        print(f'cw={cw}: best {c[b]:.3f} (seed {AE_SEEDS[b]}), median {np.median(c):.3f}, '
              f'mean {c.mean():.3f} +/- {c.std():.3f}, >0.9: {int((c > 0.9).sum())}/12, '
              f'>0.5: {int((c > 0.5).sum())}/12, seed 42: {c[0]:.3f}')


if __name__ == '__main__':
    main()
