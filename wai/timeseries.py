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
    provides = ['high level', 'low level', 'amplitude', 'rms', 'peak-peak', 'mean', 'std']
    requires = ['histogram']

    def measure(self, data, state, configuration):
        from numpy import sqrt, mean, square
        h = list(zip(*state['histogram']))

        bin_step = h[1][1] - h[0][1]

        split = int(len(h) / 2)
        bottom = h[:split]
        top = h[split:]

        state['mean'] = numpy.average(data)
        state['std'] = numpy.std(data)

        state['low level'] = max(bottom)[1] + bin_step / 2
        state['high level'] = max(top)[1] + bin_step / 2
        state['amplitude'] = state['high level'] - state['low level']
        state['peak-peak'] = max(data[0]) - min(data[0])
        state['rms'] = sqrt(mean(square(data[0])))


class Shoot(_TimeSeriesMeasurator):
    provides = ['overshoot', 'undershoot']
    requires = ['high level', 'low level']

    def measure(self, data, state, configuration):
        state['overshoot'] = max(data[0]) - state['high level']
        state['undershoot'] = min(data[0]) - state['low level']


class Edges(_TimeSeriesMeasurator):
    provides = ['rising edge', 'falling edge', 'rising edge idx', 'falling edge idx',
        'rise time', 'fall time', 'rise time std', 'fall time std',
        'rise rate', 'fall rate', 'rise rate std', 'fall rate std']
    requires = ['high level', 'low level']

    def measure(self, data, state, configuration):
        low_thres = state['low level'] + 0.1 * (state['high level'] - state['low level'])
        high_thres = state['low level'] + 0.9 * (state['high level'] - state['low level'])

        edge_height = high_thres - low_thres

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

        rise_time = numpy.array(rise_time)
        fall_time = numpy.array(fall_time)

        state['rise time'], state['rise time std'] = (numpy.average(rise_time), numpy.std(rise_time)) if len(rise_time) else ([], [])
        state['fall time'], state['fall time std'] = (numpy.average(fall_time), numpy.std(fall_time)) if len(fall_time) else ([], [])

        state['rise rate'], state['rise rate std'] = (numpy.average(edge_height / rise_time), numpy.std(edge_height / rise_time)) if len(rise_time) else ([], [])
        state['fall rate'], state['fall rate std'] = (numpy.average(edge_height / fall_time), numpy.std(edge_height / fall_time)) if len(fall_time) else ([], [])


class CycleStatistics(_TimeSeriesMeasurator):
    provides = ['cycle mean', 'cycle std', 'cycle rms']
    requires = ['rising edge idx', 'falling edge idx']

    def measure(self, data, state, configuration):
        from numpy import sqrt, mean, square
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
        state['cycle rms'] = sqrt(mean(square(trimmed)))

class PulseStatistics(_TimeSeriesMeasurator):
    provides = ['pos width', 'neg width', 'duty']
    requires = ['rising edge', 'falling edge']

    def measure(self, data, state, configuration):
        re = list(state['rising edge'])
        fe = list(state['falling edge'])

        pos = []
        neg = []

        while len(re) and len(fe):
            if re[0] < fe[0]:
                pos.append(fe[0] - re[0])
                re.pop(0)
            else:
                neg.append(re[0] - fe[0])
                fe.pop(0)

        state['pos width'] = numpy.average(pos) if len(pos) else None
        state['neg width'] = numpy.average(neg) if len(neg) else None
        state['duty'] = state['pos width'] / (state['pos width'] + state['neg width']) if state['pos width'] and state['neg width'] else None

class CycleParameters(_TimeSeriesMeasurator):
    provides = ['frequency', 'period']
    requires = ['rising edge', 'falling edge']

    def measure(self, data, state, configuration):
        re = state['rising edge']
        fe = state['falling edge']

        pr, pf = None, None
        if len(re) >= 2:
            pr = numpy.average([ b - a for a, b in zip(re, re[1:]) ])

        if len(fe) >= 2:
            pf = numpy.average([ b - a for a, b in zip(fe, fe[1:]) ])

        p = (pr + pf) / 2 if pr and pf else pr or pf

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
    def __init__(self, data, configuration={}):
        config = {}
        config.update(default_configuration)
        config.update(configuration)

        super(TimeSeriesMeasurementSet, self).__init__(
            _TimeSeriesMeasurator,
            data,
            config)

def measure(measurement, data, configuration={}):
    return TimeSeriesMeasurementSet(data, configuration).measure([measurement])[measurement]
