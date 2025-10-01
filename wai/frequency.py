import numpy

from math import floor, ceil, sqrt, pi

from wai._base import _Measurator, _BaseMeasurementSet

default_configuration = {
    'histogram bins' : 'sqrt',
    'peak method' : 'cwt',
    'peak ratio' : 10,
    'peak width' : 5,
    'units' : 'Vpp',
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
    i = interp1d(x, -y, kind=kind)

    m = minimize(i, start, method='SLSQP', bounds=[(x[0] + bw / 2, x[-1] - bw / 2)])
    return (m.x[0], -i(m.x[0]))

class _FrequencyMeasurator(_Measurator):
    pass

class LinearLogData(_FrequencyMeasurator):
    provides = ['data', 'log data']
    requires = []
    def measure(self, data, state, configuration):
        # Pre-conditions:
        # check there are negatives in dBm or dBVpp
        # check there are no negatives in Vrms or Vpp
        if configuration['units'].startswith('dB') and not any(val < 0 for val in data[0]):
            import warnings
            warnings.warn('No negative values in logarithmic data set, check units of data are dB')
        if configuration['units'].startswith('V') and not all(val >= 0 for val in data[0]):
            assert False, 'Negative values in linear data set, check config units'

        if configuration['units'].startswith('V'):
            state['data'] = data[0]
            assert data[0].dtype == float, 'Values are integers'
            data[0][(data[0] == 0.0)] = numpy.nextafter(0, 1)
            state['log data'] = numpy.log(data[0])
        if configuration['units'].startswith('dB'):
            state['data'] = numpy.exp(data[0])
            state['log data'] = data[0]

class Histogram(_FrequencyMeasurator):
    provides = ['histogram', 'log histogram']
    requires = ['data', 'log data']

    def measure(self, data, state, configuration):
        bins = configuration['histogram bins']
        state['histogram'] = numpy.histogram(state['data'], bins=bins)
        state['log histogram'] = numpy.histogram(state['log data'], bins=bins)

class NoiseFloor(_FrequencyMeasurator):
    provides = ['noise floor V', 'noise floor dB']
    requires = ['log histogram']

    def measure(self, data, state, configuration):
        from scipy.optimize import minimize
        from scipy.interpolate import interp1d

        # Generate an X-axis of bin centres rather than edges
        h = (state['log histogram'][0], state['log histogram'][1])
        bin_width = h[1][1] - h[1][0]
        xs = h[1][:-1] + bin_width / 2
        start = h[1][numpy.argmax(h[0])]
        noise_floor = _interp_max(xs, h[0], start)[0]
        state['noise floor V'] = numpy.exp(noise_floor)
        state['noise floor dB'] = numpy.log(state['noise floor V'])

class PeakLevel(_FrequencyMeasurator):
    provides = ['peak level V', 'peak level dB', 'peak frequency', 'peak idx']
    requires = ['data']

    def measure(self, data, state, configuration):
        state['peak idx'] = numpy.argmax(state['data'])
        state['peak frequency'], state['peak level V'] = _interp_max(data[1], state['data'], start=data[1][state['peak idx']])
        state['peak level dB'] = numpy.log(state['peak level V'])

class PeakSNR(_FrequencyMeasurator):
    provides = ['snr V', 'snr dB']
    requires = ['peak level V', 'noise floor V', 'peak level dB', 'noise floor dB']

    def measure(self, data, state, configuration):
        state['snr V'] = state['peak level V'] / state['noise floor V']
        state['snr dB'] = numpy.log(state['snr V'])


class OccupiedBW(_FrequencyMeasurator):
    provides = ['occupied bw']
    requires = ['data']

    def measure(self, data, state, configuration):
        from numpy import square, cumsum
        from scipy.interpolate import interp1d
        cum_power = cumsum(square(state['data']))

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
    provides = ['peak freqs', 'peak idxs', 'peak amplitudes V', 'peak amplitudes dB', 'peak snrs V', 'peak snrs dB']
    requires = ['data', 'noise floor V']

    def by_ratio(self, data, state, configuration):
        from scipy.signal import argrelmax
        threshold = configuration['peak ratio'] * state['noise floor V']

        candidates = argrelmax(state['data'])[0]
        state['peak idxs'] = [ p for p in candidates if state['data'][p] > threshold ]

    def by_cwt(self, data, state, configuration):
        from scipy.signal import find_peaks_cwt
        w = configuration['peak width']
        if isinstance(w, int):
            w = numpy.arange(1, w)

        state['peak idxs'] = find_peaks_cwt(state['data'], w)

    def measure(self, data, state, configuration):
        method = configuration['peak method']
        if method == 'ratio':
            self.by_ratio(data, state, configuration)
        elif method == 'cwt':
            self.by_cwt(data, state, configuration)
        else:
            raise Exception("Unknown peak detection method %s" % method)

        if len(state['peak idxs']):
            state['peak freqs'], state['peak amplitudes V'] = zip(*[ _interp_max(data[1], state['data'], start=s) for s in data[1][state['peak idxs']]])
            state['peak amplitudes dB'] = [ _interp_max(data[1], state['log data'], start=s)[1] for s in data[1][state['peak idxs']]]
            state['peak snrs V'] = [a / state['noise floor V'] for a in state['peak amplitudes V']]
            state['peak snrs dB'] = [a - state['noise floor dB'] for a in state['peak amplitudes dB']]
        else:
            state['peak freqs'], state['peak amplitudes V'], state['peak snrs V'] = None, None, None
            state['peak amplitudes dB'], state['peak snrs dB'] = None, None


class PeakWidths(_FrequencyMeasurator):
    provides = ['peak 3dB', 'peak 6dB', 'peak occupied']
    requires = ['peak idxs', 'peak amplitudes V', 'noise floor V']

    def _width(self, data, idx, threshold):
        from scipy.interpolate import interp1d
        from scipy.optimize import minimize

        i = interp1d(data[1], data[0], kind='cubic')
        left = minimize(lambda x: (i(x) - threshold)**2, idx - 2)
        right = minimize(lambda x: (i(x) - threshold)**2, idx + 2)

        return right.x - left.x

    def measure(self, data, state, configuration):
        state['peak 3dB'] = [ self._width(data, i, p / 2) for i, p in zip(state['peak idxs'], state['peak amplitudes V']) ]
        state['peak 6dB'] = [ self._width(data, i, p / 4) for i, p in zip(state['peak idxs'], state['peak amplitudes V']) ]
        state['peak occupied'] = [ self._width(data, i, state['noise floor V']) for i in state['peak idxs'] ]

class Distortion(_FrequencyMeasurator):
    provides = ['thd']
    requires = ['data', 'peak frequency', 'peak amplitudes V', 'peak level V']

    def measure(self, data, state, configuration):
        from scipy.interpolate import interp1d
        from scipy.optimize import minimize
        signal_power = state['peak level V']**2 # power = volts^2
        f = 2 * state['peak frequency']
        harmonic_power = 0

        i = interp1d(data[1], state['data'])

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
