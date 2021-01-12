import time


class TimeDelta:
    start_sec = None
    end_sec = None

    def start(self):
        self.start_sec = time.time()
        return self

    def end(self):
        self.end_sec = time.time()
        return self

    def delta(self):
        return self.end_sec - self.start_sec


class ProcessMeasure:
    def __init__(self):
        self.deltas = {}
        self.points = {}
        self.suffix = ''

    def set_suffix(self, suffix):
        self.suffix = suffix

    def start(self, name=None):
        self.deltas[name + self.suffix] = TimeDelta().start()

    def stop(self, name=None):
        self.data_point(self.deltas[name + self.suffix].end().delta(), collection='delta_{}'.format(name))

    def data_point(self, data_point, collection=None):
        self.points[collection + self.suffix] = self.points.get(collection + self.suffix, [])
        self.points[collection + self.suffix].append(data_point)

    def print(self):
        print('{')
        for key in self.points.keys():
            print("\t'{}': {}".format(key, self.points[key]))
        print('}')