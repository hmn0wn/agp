import time
from collections import OrderedDict

class TimerPerf:
    def __init__(self):
        self._prev_time = None
        self._laps = OrderedDict()
        self._unnamed_num = 0

    def get(self, name=None):
        assert(name)
        return self._laps[name]


    def lap(self, name=None, is_print=False):
        if not self._prev_time:
            self._prev_time = time.perf_counter()
            return 0

        elapsed_time = time.perf_counter() - self._prev_time
        self._prev_time = time.perf_counter()

        if not name:
            self._laps[str(self._unnamed_num)] = elapsed_time
            self._unnamed_num += 1
            return elapsed_time

        if name in self._laps:
            self._laps[name] += elapsed_time
        else:
            self._laps[name] = elapsed_time

        return elapsed_time

    def merge(self, timer):
        for key,value in timer._laps.items():
            if key in self._laps:
                self._laps[key] += value
            else:
                self._laps[key] = value

    def total(self):
        sum = 0
        for key, value in self._laps.items():
            sum += value
        return sum
            
    def print(self):
        print("-"*20)
        for key, value in self._laps.items():
            print(f"{key}: {value:>10.4f}")
        print("-"*20)