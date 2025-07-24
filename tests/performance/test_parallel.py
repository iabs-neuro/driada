import numpy as np
from driada.intense.intense_base import (calculate_optimal_delays_parallel,
                                             calculate_optimal_delays,
                                             scan_pairs,
                                             scan_pairs_parallel,
                                             scan_pairs_router)

from driada.information.info_base import TimeSeries

shift_window = 40
ds = 5

def test_mi_equality():
    ts_bunch1 = [TimeSeries(np.random.random(size=1000)) for _ in range(12)]
    ts_bunch2 = [TimeSeries(np.random.random(size=1000)) for _ in range(20)]
    optd = np.random.randint(-40, 40, size=(12, 20))

    rshifts1, mitable1 = scan_pairs(ts_bunch1,
                                    ts_bunch2,
                                    metric='mi',
                                    nsh=100,
                                    optimal_delays=optd,
                                    ds=5,
                                    joint_distr=False,
                                    noise_const=1e-4,
                                    seed=42)

    rshifts2, mitable2 = scan_pairs_parallel(ts_bunch1,
                                             ts_bunch2,
                                             'mi',
                                             100,
                                             optd,
                                             ds=5,
                                             joint_distr=False,
                                             n_jobs=-1,
                                             noise_const=1e-4,
                                             seed=42)

    print(rshifts1.shape)
    print(rshifts2.shape)
    assert np.allclose(rshifts1, rshifts2)
    assert np.allclose(mitable1, mitable2)


def test_wrapper():
    ts_bunch1 = [TimeSeries(np.random.random(size=1000)) for _ in range(12)]
    ts_bunch2 = [TimeSeries(np.random.random(size=1000)) for _ in range(20)]
    optd = np.random.randint(-40, 40, size=(12, 20))

    rshifts1, mitable1 = scan_pairs_router(ts_bunch1,
                                           ts_bunch2,
                                           'mi',
                                           100,
                                           optd,
                                           ds=5,
                                           joint_distr=False,
                                           noise_const=1e-3,
                                           seed=42,
                                           enable_parallelization=True,
                                           n_jobs=-1)

    rshifts2, mitable2 = scan_pairs_router(ts_bunch1,
                                           ts_bunch2,
                                           'mi',
                                           100,
                                           optd,
                                           ds=5,
                                           joint_distr=False,
                                           noise_const=1e-3,
                                           seed=42,
                                           enable_parallelization=False,
                                           n_jobs=-1)

    print(rshifts1.shape)
    print(rshifts2.shape)
    assert np.allclose(rshifts1, rshifts2)
    assert np.allclose(mitable1, mitable2)


def test_delays_equality():
    ts_bunch1 = [TimeSeries(np.random.random(size=1000)) for _ in range(12)]
    ts_bunch2 = [TimeSeries(np.random.random(size=1000)) for _ in range(20)]

    optimal_delays1 = calculate_optimal_delays_parallel(ts_bunch1,
                                                        ts_bunch2,
                                                        shift_window,
                                                        ds,
                                                        verbose=False,
                                                        n_jobs=-1)

    optimal_delays2 = calculate_optimal_delays(ts_bunch1,
                                               ts_bunch2,
                                               shift_window,
                                               ds,
                                               verbose=False)

    print(optimal_delays1.shape)
    print(optimal_delays2.shape)
    assert np.allclose(optimal_delays1, optimal_delays2)