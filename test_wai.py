#!/usr/bin/env python3

from wai.timeseries import TimeSeriesMeasurementSet

from pylab import *

import scipy.signal
import numpy
import time

from math import pi

t = numpy.linspace(0, 1, 1000)
n = numpy.random.normal(scale=0.1, size=t.shape)
#data = [scipy.signal.square(t * 5 * 2 * pi) + n, t]
data = [numpy.sin(t*5*2*pi) + n, t]
m = TimeSeriesMeasurementSet()

start = time.clock()
ms = m.measure(data)
print(time.clock() - start)
print(ms)

plot(data[0],'.')
plot(numpy.sin(ms['frequency'] * t * 2 * pi))
plot(ms['sine offset'] +
    ms['sine amplitude'] *
        numpy.sin(ms['sine frequency'] * t * 2 * pi + ms['sine phase']))
show()
