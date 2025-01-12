"""
Implementation From:
- Piecewise: https://github.com/labmlai/labml/blob/master/helpers/labml_helpers/schedule.py
"""

from typing import List, Tuple


class Piecewise:
    def __init__(self, endpoints: List[Tuple[float, float]], outside_value: float):
        """
        `endpoints` is list of pairs `(x, y)`.
         The values between endpoints are linearly interpolated.
        `y` values outside the range covered by `x` are
        `outside_value`.
        """

        # `(x, y)` pairs should be sorted
        indexes = [e[0] for e in endpoints]
        assert indexes == sorted(indexes)

        self._outside_value = outside_value
        self._endpoints = endpoints

    def __call__(self, x):
        """
        Find `y` for given `x` (timestep)
        """

        # iterate through each segment
        for (x1, y1), (x2, y2) in zip(self._endpoints[:-1], self._endpoints[1:]):
            # interpolate if `x` is within the segment
            if x1 <= x < x2:
                dx = float(x - x1) / (x2 - x1)
                return y1 + dx * (y2 - y1)

        # return outside value otherwise
        return self._outside_value
