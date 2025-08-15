import numpy as np
from ..utils.jit import conditional_njit, is_jit_enabled

# Import numba types only if JIT is enabled
if is_jit_enabled():
    from numba.experimental import jitclass
    from numba import float32, boolean
    from numba import types, typed
else:
    # Provide dummy implementations
    jitclass = lambda spec: lambda cls: cls
    types = None
    typed = None


# Define spec only when numba is available
if is_jit_enabled():
    spec = [
        ("indices", types.ListType(types.float64)),
        ("ampls", types.ListType(types.float64)),
        ("birth_scale", float32),
        ("scales", types.ListType(types.float64)),
        ("wvt_times", types.ListType(types.float64)),
        ("terminated", boolean),
        ("end_scale", float32),
        ("length", float32),
        ("max_scale", float32),
        ("max_ampl", float32),
        ("start", float32),
        ("end", float32),
        ("duration", float32),
    ]
else:
    spec = None


@conditional_njit()
def maxpos_numba(x):
    m = max(x)
    # Handle both list and typed.List
    if hasattr(x, 'index'):
        return x.index(m)
    else:
        # Fallback for regular lists
        for i, val in enumerate(x):
            if val == m:
                return i
        return 0


# Conditional compilation for Ridge class
if is_jit_enabled():
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
                raise ValueError("Ridge is terminated")

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
                self.duration = np.abs(self.end - self.start)
                self.terminated = True
else:
    # Pure Python implementation
    class Ridge(object):
        def __init__(self, start_index, ampl, start_scale, wvt_time):
            self.indices = [start_index]
            self.ampls = [ampl]
            self.birth_scale = start_scale
            self.scales = [start_scale]
            self.wvt_times = [wvt_time]
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
                raise ValueError("Ridge is terminated")

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
                self.duration = np.abs(self.end - self.start)
                self.terminated = True


class RidgeInfoContainer(object):
    def __init__(self, indices, ampls, scales, wvt_times):
        self.indices = np.array(indices)
        self.ampls = np.array(ampls)
        self.scales = np.array(scales)
        self.wvt_times = np.array(wvt_times)

        self.birth_scale = scales[0]
        self.end_scale = scales[-1]
        self.length = len(self.scales)
        self.max_scale = self.scales[np.argmax(self.ampls)]
        self.max_ampl = max(self.ampls)
        self.start = self.indices[0]
        self.end = self.indices[-1]
        self.duration = np.abs(self.end - self.start)


def ridges_to_containers(ridges):
    rcs = [
        RidgeInfoContainer(ridge.indices, ridge.ampls, ridge.scales, ridge.wvt_times)
        for ridge in ridges
    ]
    return rcs
