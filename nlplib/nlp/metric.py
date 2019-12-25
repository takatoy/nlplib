class Metric:
    def __init__(self, name):
        self.name = name

    def get(self, output, label):  # pragma: no cover
        pass


class Accuracy(Metric):
    def __init__(self, name, key):
        super().__init__(name)
        self.key = key
        self.correct = 0
        self.total = 0

    def get(self, output, label):
        self.correct, self.total = self._get(output[self.key], label[self.key])
        return self.correct / self.total

    def _get(self, output, label):
        correct = 0
        total = 0
        if isinstance(output, list):
            assert isinstance(label, list)
            for o, l in zip(output, label):
                c, t = self._get(o, l)
                correct += c
                total += t
            total += max(len(output), len(label)) - min(len(output), len(label))
            return correct, total
        return int(output == label), 1
