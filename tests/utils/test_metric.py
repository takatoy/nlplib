import pytest
import nlplib.utils.metric as metric


class TestAccuracy:
    def setup_method(self, method):
        self.accuracy = metric.Accuracy("test", "val")

    def test_get(self):
        x = {"val": [2, 7, 3, 10, 25, 198, 2976, 13, 28, 10]}
        y = {"val": [2, 7, 0, 10, 25, 197, 2222, 89, 28, 15]}
        assert pytest.approx(self.accuracy.get(x, y)) == 0.5

        x = {"val": [[2, 3, 8], [12, 28, 19, 21], [9, 10]]}
        y = {"val": [[2, 3, 8, 8], [11, 29, 19], [8, 10]]}
        assert pytest.approx(self.accuracy.get(x, y)) == 0.5
