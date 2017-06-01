import numpy

from math import floor, ceil, sqrt, pi

from wai._base import _Measurator, _BaseMeasurementSet

default_configuration = {
    'histogram bins' : 'sqrt',
}


class _TimeSeriesMeasurator(_Measurator):
    pass


class Histogram(_TimeSeriesMeasurator):
    provides = ['histogram']
    requires = []

    def measure(self, data, state, configuration):
        bins = configuration['histogram bins']
        state['histogram'] = numpy.histogram(data[0], bins=bins)
        

class Levels(_TimeSeriesMeasurator):
    provides = ['high level', 'low level', 'amplitude']
    requires = ['histogram']

    def measure(self, data, state, configuration):
        h = list(zip(*state['histogram']))

        bin_step = h[1][1] - h[0][1]

        split = int(len(h) / 2)
        bottom = h[:split]
        top = h[split:]

        state['low level'] = max(bottom)[1] + bin_step / 2
        state['high level'] = max(top)[1] + bin_step / 2
        state['amplitude'] = (state['high level'] - state['low level']) / 2


class Shoot(_TimeSeriesMeasurator):
    provides = ['overshoot', 'undershoot']
    requires = ['high level', 'low level']

    def measure(self, data, state, configuration):
        state['overshoot'] = max(data[0]) - state['high level']
        state['undershoot'] = min(data[0]) - state['low level']


class Edges(_TimeSeriesMeasurator):
    provides = ['rising edge', 'falling edge', 'rising edge idx', 'falling edge idx', 'rise time', 'fall time', 'rise time std', 'fall time std']
    requires = ['high level', 'low level']

    def measure(self, data, state, configuration):
        low_thres = state['low level'] + 0.1 * (state['high level'] - state['low level'])
        high_thres = state['low level'] + 0.9 * (state['high level'] - state['low level'])

        rising_points = []
        falling_points = []

        detect_state = 'start'
        last = (0,0)
        for i, d1, d2, t1, t2 in zip(range(len(data[0])), data[0], data[0][1:], data[1], data[1][1:]):
            if d1 <= low_thres and d2 > low_thres and detect_state in ['rising low', 'start']:
                r = (d2 - low_thres) / (d2 - d1)
                t = r * t2 + (1 - r) * t1
                last = (i,t)
                detect_state = 'rising high'
            if d1 <= high_thres and d2 > high_thres and detect_state in ['rising high']:
                r = (d2 - high_thres) / (d2 - d1)
                t = r * t2 + (1 - r) * t1
                li, lt = last
                rising_points.append((int((i + li) / 2), (t + lt) / 2, t - lt))
                detect_state = 'falling high'
            if d1 > high_thres and d2 <= high_thres and detect_state in ['falling high', 'start']:
                r = (d2 - high_thres) / (d2 - d1)
                t = r * t2 + (1 - r) * t1
                last = (i,t)
                detect_state = 'falling low'
            if d1 > low_thres and d2 <= low_thres and detect_state in ['falling low']:
                r = (d2 - low_thres) / (d2 - d1)
                t = r * t2 + (1 - r) * t1
                li, lt = last
                falling_points.append((int((i + li) / 2), (t + lt) / 2, t - lt))
                detect_state = 'rising low'
        
        state['rising edge idx'], state['rising edge'], rise_time = zip(*rising_points) if len(rising_points) else ([], [], [])
        state['falling edge idx'], state['falling edge'], fall_time = zip(*falling_points) if len(falling_points) else ([], [], [])

        state['rise time'], state['rise time std'] = (numpy.average(rise_time), numpy.std(rise_time)) if len(rise_time) else ([], [])
        state['fall time'], state['fall time std'] = (numpy.average(fall_time), numpy.std(fall_time)) if len(fall_time) else ([], [])

class Statistics(_TimeSeriesMeasurator):
    provides = ['mean', 'std', 'cycle mean', 'cycle std']
    requires = ['rising edge idx', 'falling edge idx']

    def measure(self, data, state, configuration):
        state['mean'] = numpy.average(data)
        state['std'] = numpy.std(data)

        rei = state['rising edge idx']
        fei = state['falling edge idx']

        if len(rei) >= 2:
            trimmed = data[0][rei[0]:rei[-1]]
        elif len(fei) >= 2:
            trimmed = data[0][fei[0]:fei[-1]]
        else:
            trimmed = []
        
        state['cycle mean'] = numpy.average(trimmed)
        state['cycle std'] = numpy.std(trimmed)


class CycleParameters(_TimeSeriesMeasurator):
    provides = ['frequency', 'period']
    requires = ['rising edge', 'falling edge']

    def measure(self, data, state, configuration):
        re = state['rising edge']
        fe = state['falling edge']

        if len(re) >= 2:
            p = numpy.average([ b - a for a, b in zip(re, re[1:]) ])
        elif len(fe) >= 2:
            p = numpy.average([ b - a for a, b in zip(fe, fe[1:]) ])
        else:
            p = None
        
        f = 1 / p if p else None

        state['frequency'] = f
        state['period'] = p


class SineParameters(_TimeSeriesMeasurator):
    provides = ['sine frequency', 'sine offset', 'sine phase', 'sine amplitude']
    requires = ['frequency', 'period', 'cycle mean', 'amplitude', 'rising edge']

    def measure(self, data, state, configuration):
        from scipy.optimize import curve_fit

        def sine_objective(x, freq, amplitude, phase, offset):
            return numpy.sin(x * freq * 2 * pi + phase) * amplitude + offset
        
        # Don't try and fit a sine if we don't have an estimated period, i.e. at least
        # one complete cycle
        if state['period']:
            est_phase = state['rising edge'][0] / state['period'] * 2 * pi
        else:
            state['sine frequency'], state['sine amplitude'], state['sine phase'], state['sine offset'] = None, None, None, None
            return

        p0 = [state['frequency'], state['amplitude'], est_phase, state['cycle mean']]

        try:
            fit = curve_fit(sine_objective, data[1], data[0], p0)[0]
            if fit[1] < 0:
                fit[2] += pi
                fit[1] *= -1
        except RuntimeError:
            fit = None, None, None, None

        state['sine frequency'], state['sine amplitude'], state['sine phase'], state['sine offset'] = fit

class TimeSeriesMeasurementSet(_BaseMeasurementSet):
    def __init__(self, measurements='all', configuration={}):
        config = {}
        config.update(default_configuration)
        config.update(configuration)

        super(TimeSeriesMeasurementSet, self).__init__(
            _TimeSeriesMeasurator,
            measurements,
            config)
