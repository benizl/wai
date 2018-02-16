import numpy

from math import floor, ceil, sqrt, pi

from wai._base import _Measurator, _BaseMeasurementSet

default_configuration = {
    'histogram bins' : 'sqrt',
    'peak method' : 'cwt',
    'peak ratio' : 10,
    'peak width' : 5,
}

def _interp_max(x, y, start=None):
    from scipy.optimize import minimize
    from scipy.interpolate import interp1d

    if len(x) < 3 or len(y) < 3:
        kind = 'linear'
    else:
        kind = 'cubic'

    start = start or x[int(len(x)/2)]
    bw = x[1] - x[0]
    print(x[0], x[1], x[-1])
    i = interp1d(x, -y, kind='linear')
    def j(x):
        print(x)
        return i(x)

    m = minimize(j, start, method='SLSQP', bounds=[(x[0] + bw / 2, x[-1] - bw / 2)])
    return (m.x[0], -m.fun[0])

class _FrequencyMeasurator(_Measurator):
    pass


class Histogram(_FrequencyMeasurator):
    provides = ['histogram', 'log histogram']
    requires = []

    def measure(self, data, state, configuration):
        bins = configuration['histogram bins']
        state['histogram'] = numpy.histogram(data[0], bins=bins)

        count, amp = numpy.histogram(numpy.log(data[0]), bins=bins)
        amp = numpy.exp(amp)

        state['log histogram'] = numpy.array([count, amp])

class NoiseFloor(_FrequencyMeasurator):
    provides = ['noise floor']
    requires = ['histogram']

    def measure(self, data, state, configuration):
        from scipy.optimize import minimize
        from scipy.interpolate import interp1d

        # Generate an X-axis of bin centres rather than edges
        h = state['log histogram']
        bin_width = h[1][1] - h[1][0]
        xs = h[1][:-1] + bin_width / 2
        state['noise floor'] = _interp_max(xs, h[0])[0]

class PeakLevel(_FrequencyMeasurator):
    provides = ['peak level', 'peak frequency', 'peak idx']
    requires = []

    def measure(self, data, state, configuration):
        state['peak idx'] = numpy.argmax(data[0])
        state['peak frequency'], state['peak level'] = _interp_max(data[1], data[0], start=state['peak idx'])

class PeakSNR(_FrequencyMeasurator):
    provides = ['snr']
    requires = ['peak level', 'noise floor']

    def measure(self, data, state, configuration):
        state['snr'] = state['peak level'] / state['noise floor']


class OccupiedBW(_FrequencyMeasurator):
    provides = ['occupied bw']
    requires = []

    def measure(self, data, state, configuration):
        from numpy import square, cumsum
        from scipy.interpolate import interp1d
        cum_power = cumsum(square(data[0]))

        total_power = cum_power[-1]
        low_thresh = 0.005 * total_power
        high_thresh = 0.995 * total_power

        # Note the power has become our 'x' axis so we can directly evaluate at our
        # threshold powers.
        i = interp1d(cum_power, data[1], kind='linear')
        f_low = i(low_thresh)
        f_high = i(high_thresh)

        state['occupied bw'] = f_high - f_low

class Peaks(_FrequencyMeasurator):
    provides = ['peak freqs', 'peak idxs', 'peak amplitudes', 'peak snrs']
    requires = ['noise floor']

    def by_ratio(self, data, state, configuration):
        from scipy.signal import argrelmax
        threshold = configuration['peak ratio'] * state['noise floor']

        candidates = argrelmax(data[0])[0]
        state['peak idxs'] = [ p for p in candidates if data[0][p] > threshold ]

    def by_cwt(self, data, state, configuration):
        from scipy.signal import find_peaks_cwt
        w = configuration['peak width']
        if isinstance(w, int):
            w = numpy.arange(1, w)

        state['peak idxs'] = find_peaks_cwt(data[0], w)

    def measure(self, data, state, configuration):
        method = configuration['peak method']
        if method == 'ratio':
            self.by_ratio(data, state, configuration)
        elif method == 'cwt':
            self.by_cwt(data, state, configuration)
        else:
            raise Exception("Unknown peak detection method %s" % method)

        if len(state['peak idxs']):
            state['peak freqs'], state['peak amplitudes'] = zip(*[ _interp_max(data[1], data[0], start=s) for s in data[1][state['peak idxs']]])
            state['peak snrs'] = [a / state['noise floor'] for a in state['peak amplitudes']]
        else:
            state['peak freqs'], state['peak amplitudes'], state['peak snrs'] = None, None, None


def PeakWidths(_FrequencyMeasurator):
    provides = ['peak 3dB', 'peak 6dB', 'peak occupied']
    requires = ['peak idxs', 'peak amplitudes', 'noise floor']

    def _width(self, data, idx, threshold):
        from scipy.interpolate import interp1d

        i = interp1d(data[1], data[0], kind='cubic')
        left = minimize(lambda x: (i(x) - threshold)**2, idx - 2)
        right = minimize(lambda x: (i(x) - threshold)**2, idx + 2)

        return right.x - left.x

    def measure(self, data, state, configuration):
        state['peak 3dB'] = [ self._width(data, i, p / 2) for i, p in zip(state['peak idxs'], state['peak amplitudes']) ]
        state['peak 6dB'] = [ self._width(data, i, p / 4) for i, p in zip(state['peak idxs'], state['peak amplitudes']) ]
        state['peak occupied'] = [ self._width(data, i, state['noise floor']) for i in state['peak idxs'] ]

def Distortion(_FrequencyMeasurator):
    provides = ['thd']
    requires = ['peak frequency', 'peak amplitude']

    def measure(self, data, state, configuration):
        from scipy.interpolate import interp1d
        signal_power = state['peak amplitude']**2 # power = volts^2
        f = 2 * state['peak frequency']
        harmonic_power = 0

        i = interp1d(data[1], data[0])

        while f < data[1][-1]:
            harmonic_power += i(f)**2
            f += state['peak frequency']

        if harmonic_power == 0:
            state['thd'] = None
        else:
            state['thd'] = signal_power / harmonic_power


class FrequencyMeasurementSet(_BaseMeasurementSet):
    def __init__(self, data, configuration={}):
        config = {}
        config.update(default_configuration)
        config.update(configuration)

        super(FrequencyMeasurementSet, self).__init__(
            _FrequencyMeasurator,
            data,
            config)

def measure(measurement, data, configuration={}):
    return FrequencyMeasurementSet(data, configuration).measure([measurement])[measurement]
