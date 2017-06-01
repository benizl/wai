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
    provides = ['rising edge', 'falling edge', 'rising edge idx', 'falling edge idx']
    requires = ['high level', 'low level']

    def measure(self, data, state, configuration):
        centre = (state['high level'] - state['low level']) / 2 + state['low level']

        state['rising edge'] = []
        state['falling edge'] = []

        state['rising edge idx'] = []
        state['falling edge idx'] = []

        for i, d1, d2, t1, t2 in zip(range(len(data[0])), data[0], data[0][1:], data[1], data[1][1:]):
            if d1 <= centre and d2 > centre:
                r = (d2 - centre) / (d2 - d1)
                t = r * t2 + (1 - r) * t1
                state['rising edge'].append(t)
                state['rising edge idx'].append(i)
            elif d1 > centre and d2 <= centre:
                r = (d2 - centre) / (d2 - d1)
                t = r * t2 + (1 - r) * t1
                state['falling edge'].append(t)
                state['falling edge idx'].append(i)


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
        


class EdgeRates(_TimeSeriesMeasurator):
    provides = ['rise time', 'fall time', 'rise time std', 'fall time std']
    requires = ['high level', 'low level', 'rising edge idx', 'falling edge idx']

    def __init__(self):
        self._risetimes = []
        self._falltimes = []

    def measure(self, data, state, configuration):
        low_thres = state['low level'] + 0.1 * (state['high level'] - state['low level'])
        high_thres = state['low level'] + 0.9 * (state['high level'] - state['low level'])

        for edge in state['rising edge idx']:
            # This zip/reversed thing looks ugly, but having it in this order
            # keeps everything as generators rather than having to actually
            # build the full list, speeding up execution and helping memory
            lt = ht = None
            for d1, d2, t1, t2 in zip(reversed(data[0][:edge+1]), reversed(data[0][2:edge+2]), reversed(data[1][1:edge+1]), reversed(data[1][2:edge+2])):
                if d1 <= low_thres and d2 > low_thres:
                    r = (d2 - low_thres) / (d2 - d1)
                    lt = r * t2 + (1 - r) * t1
                    break
            
            for d1, d2, t1, t2 in zip(data[0][edge:], data[0][edge+1:], data[1][edge:], data[1][edge+1:]):
                if d1 <= high_thres and d2 > high_thres:
                    r = (d2 - high_thres) / (d2 - d1)
                    ht = r * t2 + (1 - r) * t1
                    break
            
            if ht is not None and lt is not None:
                self._risetimes.append(lt - ht)

        for edge in state['falling edge idx']:
            lt = ht = None
            for d1, d2, t1, t2 in zip(data[0][edge:], data[0][edge+1:], data[1][edge:], data[1][edge+1:]):
                if d1 > low_thres and d2 <= low_thres:
                    r = (d2 - low_thres) / (d2 - d1)
                    lt = r * t2 + (1 - r) * t1
                    break
            
            for d1, d2, t1, t2 in zip(reversed(data[0][:edge+1]), reversed(data[0][2:edge+2]), reversed(data[1][1:edge+1]), reversed(data[1][2:edge+2])):
                if d1 > high_thres and d2 <= high_thres:
                    r = (d2 - high_thres) / (d2 - d1)
                    ht = r * t2 + (1 - r) * t1
                    break
            
            if ht is not None and lt is not None:
                self._falltimes.append(ht - lt)

            state['rise time'] = numpy.average(self._risetimes)
            state['fall time'] = numpy.average(self._falltimes)

            state['rise time std'] = numpy.std(self._risetimes)
            state['fall time std'] = numpy.std(self._falltimes)


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
        
        if len(state['rising edge']):
            est_phase = state['rising edge'][0] / state['period'] * 2 * pi
        else:
            est_phase = 0

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
