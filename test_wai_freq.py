#!/usr/bin/env python3

import wai.timeseries
from wai.frequency import FrequencyMeasurementSet

from pylab import *

import scipy.signal
import numpy
import time

from math import pi

t = numpy.linspace(0, 1, 1000)
fs = 1 / (t[1] - t[0])
n = numpy.random.normal(scale=0.1, size=t.shape)
data = [scipy.signal.square(t * 50 * 2 * pi) + n, t]
#data = [numpy.sin(t*50*2*pi) + n, t]
# data = [[0, 0, 1, 1, 0, 0, 1, 1], [0,1,2,3,4,5,6,7]]

f, p = scipy.signal.periodogram(data[0], fs, 'flattop', scaling='spectrum')

config = {
    'peak method': 'cwt'
}

m = FrequencyMeasurementSet([p, f], configuration=config)

start = time.clock()
ms = m.measure()
print(time.clock() - start)
print(ms)

semilogy(f, p)
semilogy(ms['peak freqs'], ms['peak amplitudes'], 'o')
show()

print(wai.timeseries.measure('rising edge', data))
