
class _Measurator(object):
    provides = []
    requires = []

    @classmethod
    def all_measurators(cls):
        for subclass in cls.__subclasses__():
            # the right move is 'yield from' but we
            # still need to support Python 2.x
            for m in subclass.all_measurators():
                yield m
            yield subclass

    def measure(self, data, state, configuration):
        pass


class _BaseMeasurementSet(object):

    def __init__(self, base_class, data, configuration=None):
        self.base_class = base_class
        self.data = data
        self.configuration = configuration

        self.state = {}


    def _deps_satisfied(self, measurement, state):
        for r in measurement.requires:
            if not r in state:
                return False

        return True

    def _provider(self, m):
        for cl in self.base_class.all_measurators():
            if m in cl.provides:
                required = set([cl])
                for req in cl.requires:
                    required |= self._provider(req)
                return required
        else:
            raise Exception("Can't find provider for {}".format(m))

    def measure(self, measurements='all'):

        if isinstance(measurements, str):
            measurements = [measurements]

        measurement_classes = set()
        for m in measurements:
            if m == 'all':
                measurement_classes |= set(self.base_class.all_measurators())
            else:
                measurement_classes |= self._provider(m)

        # Instantiate all required classes. We'll keep the objects
        # around in the future so more data can be added
        self._measurators = [ cls() for cls in measurement_classes ]

        to_measure = set(self._measurators)

        if len(self.data) != 2:
            self.data = [self.data, range(len(self.data))]

        while len(to_measure):
            processed = set()
            for try_measure in to_measure:
                if self._deps_satisfied(try_measure, self.state):
                    try_measure.measure(self.data, self.state, self.configuration)
                    processed |= set([try_measure])

            if not len(processed):
                raise Exception("Can't make progress through measurements, must be an invalid configuration somewhere")

            to_measure -= processed

        return self.state
