import time
from collections import OrderedDict

import itertools

BR_LEN = 100
BR_LENH = BR_LEN // 2
class TimerPerf:
    def __init__(self):
        self._prev_time = None
        self._laps = OrderedDict()
        self._unnamed_num = 0

    def get(self, name=None):
        assert(name)
        return self._laps[name][0]


    def lap(self, name=None, is_print=False):
        if not self._prev_time:
            self._prev_time = time.perf_counter()
            return 0

        elapsed_time = time.perf_counter() - self._prev_time
        self._prev_time = time.perf_counter()

        if not name:
            self._laps[str(self._unnamed_num)] = [elapsed_time,1]
            self._unnamed_num += 1
            return elapsed_time

        if name in self._laps:
            self._laps[name][0] += elapsed_time
        else:
            self._laps[name] = [elapsed_time,1]

        return elapsed_time

    def merge(self, timer):
        for key,value in timer._laps.items():
            if key in self._laps:
                self._laps[key][0] += value[0]
                self._laps[key][1] += value[1]
            else:
                self._laps[key] = value

    def total(self):
        sum = 0
        for key, value in self._laps.items():
            sum += value[0]
        return sum
            
    def print(self):
        #print(f"var:{type(itertools.count(0)).__name__}")
        print("-"*BR_LENH)
        for key, value in self._laps.items():
            print(f"{key:>20}: {value[0]:<10.4f} | avg: {value[0]/value[1]:<10.4f} of num: {value[1]}")
        print("="*BR_LENH)