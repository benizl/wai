#!/usr/bin/env python3

import wai.timeseries
from wai.timeseries import TimeSeriesMeasurementSet

from pylab import *

import scipy.signal
import numpy
import time

from math import pi

t = numpy.linspace(0, 1, 1000)
n = numpy.random.normal(scale=0.05, size=t.shape)
data = [scipy.signal.square(t * 5 * 2 * pi) + n, t]
# data = [numpy.sin(t*5*2*pi) + n, t]
# data = [[0, 0, 1, 1, 0, 0, 1, 1], [0,1,2,3,4,5,6,7]]
m = TimeSeriesMeasurementSet(data)

start = time.clock()
ms = m.measure()
print(time.clock() - start)
print(ms)

for r in [ ((r, r),(-1, 1)) for r in ms['rising edge idx']]:
    plot(*r, 'k')

for r in [ ((r, r),(-1, 1)) for r in ms['falling edge idx']]:
    plot(*r, 'c')

plot(data[0],'.')
plot(ms['amplitude'] / 2 * numpy.sin(ms['frequency'] * t * 2 * pi + ms['rising edge'][0]) + ms['cycle mean'])
plot(ms['sine offset'] +
    ms['sine amplitude'] *
        numpy.sin(ms['sine frequency'] * t * 2 * pi + ms['sine phase']))
show()
