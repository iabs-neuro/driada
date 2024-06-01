from numba.experimental import jitclass
import numpy as np
from numba import int32, float32, boolean    # import the types
from numba import types, typed, njit


spec = [
    ('indices', types.ListType(types.float64)),
    ('ampls', types.ListType(types.float64)),
    ('birth_scale', float32),
    ('scales', types.ListType(types.float64)),
    ('wvt_times', types.ListType(types.float64)),
    ('terminated', boolean),
    ('end_scale', float32),
    ('length', float32),
    ('max_scale', float32),
    ('max_ampl', float32),
    ('start', float32),
    ('end', float32),
    ('duration', float32),
]


@njit()
def maxpos_numba(x):
    m = max(x)
    return x.index(m)


@jitclass(spec)
class Ridge(object):

    def __init__(self, start_index, ampl, start_scale, wvt_time):
        self.indices = typed.List.empty_list(types.float64)
        self.indices.append(start_index)

        self.ampls = typed.List.empty_list(types.float64)
        self.ampls.append(ampl)

        self.birth_scale = start_scale

        self.scales = typed.List.empty_list(types.float64)
        self.scales.append(start_scale)

        self.wvt_times = typed.List.empty_list(types.float64)
        self.wvt_times.append(wvt_time)

        self.terminated = False

        self.end_scale = -1
        self.length = -1
        self.max_scale = -1
        self.max_ampl = -1
        self.start = -1
        self.end = -1
        self.duration = -1


    def extend(self, index, ampl, scale, wvt_time):
        if not self.terminated:
            self.scales.append(scale)
            self.ampls.append(ampl)
            self.indices.append(index)
            self.wvt_times.append(wvt_time)
        else:
            raise ValueError('Ridge is terminated')


    def tip(self):
        return self.indices[-1]


    def terminate(self):
        if self.terminated:
            pass

        else:
            self.end_scale = self.scales[-1]
            self.length = len(self.scales)
            self.max_scale = self.scales[maxpos_numba(self.ampls)]
            self.max_ampl = max(self.ampls)
            self.start = self.indices[0]
            self.end = self.indices[-1]
            self.duration = np.abs(self.end-self.start)
            self.terminated = True