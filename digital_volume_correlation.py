import numpy
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import scipy.optimize as opt


class DigitalVolumeCorrelation:

    def __init__(self):
        pass

    def normalised_correlation_criterion(self, windowIn1, windowIn2):
        window1 = windowIn1 - np.mean(windowIn1)
        window2 = windowIn2 - np.mean(windowIn2)

        sum1 = (window1 ** 2).sum()
        sum2 = (window2 ** 2).sum()

        ncc = ((window1 * window2).sum() / numpy.sqrt(sum1 * sum2)).astype('<f8')

        return ncc

    def normalised_least_squares_criterion(self, windowIn1, windowIn2):
        window1 = windowIn1 - np.mean(windowIn1)
        window2 = windowIn2 - np.mean(windowIn2)

        sum1 = (window1 ** 2).sum()
        sum2 = (window2 ** 2).sum()

        right = window2 / numpy.sqrt(sum2)
        if np.any(np.isnan(right)):
            right.fill(0.0)

        left = window1 / numpy.sqrt(sum1)
        if np.any(np.isnan(left)):
            left.fill(0.0)

        diffs = left - right

        nlsc = (diffs ** 2).sum()

        return nlsc

    def cross_correlation_func_trans(self, variable, original_coords, interpolated_reference, interpolated_deformed, grid, interpolationConfig):
        x, y, z = original_coords
        u, v, w = variable

        reference_window = (grid[0] + x, grid[1] + y, grid[2] + z)
        deformed_window = (grid[0] + x + u, grid[1] + y + v, grid[2] + z + w)

        window1 = interpolated_reference(reference_window, method=interpolationConfig)  # quintic
        window2 = interpolated_deformed(deformed_window, method=interpolationConfig)  # quintic

        return self.normalised_least_squares_criterion(window1, window2)

    def cross_correlation_func_full(self, variable, original_coords,interpolated_reference, interpolated_deformed, grid, interpolationConfig):
        x, y, z = original_coords
        u, v, w, dux, duy, duz, dvx, dvy, dvz, dwx, dwy, dwz = variable

        reference_window = (x + grid[0], y + grid[1], z + grid[2])
        deformed_window = (x + grid[0] + u + dux * grid[0] + duy * grid[1] + duz * grid[2],
                           y + grid[1] + v + dvx * grid[0] + dvy * grid[1] + dvz * grid[2],
                           z + grid[2] + w + dwx * grid[0] + dwy * grid[1] + dwz * grid[2])

        window1 = interpolated_reference(reference_window, method=interpolationConfig)  # quintic
        window2 = interpolated_deformed(deformed_window, method=interpolationConfig)  # quintic

        return self.normalised_least_squares_criterion(window1, window2)

    def find_correlated_point(self, reference_volume, deformed_volume, point, windowSizeConfig, onlyTranslationConfig, interpolationConfig):
        '''
        reference_volume, deformed_volume - 3D numpy array
        point - tuple (x, y, z)
        '''

        dim = reference_volume.shape

        window_size = windowSizeConfig
        window_center = window_size // 2 + 1
        window_side = window_size - window_center

        range_x = np.array(range(0, dim[0]))
        range_y = np.array(range(0, dim[1]))
        range_z = np.array(range(0, dim[2]))
        interpolated_reference = RegularGridInterpolator((range_x, range_y, range_z), reference_volume, fill_value=0.0, bounds_error=False)
        interpolated_deformed = RegularGridInterpolator((range_x, range_y, range_z), deformed_volume, fill_value=0.0, bounds_error=False)

        grid = np.meshgrid(np.array(range(-window_side, window_side + 1)),
                           np.array(range(-window_side, window_side + 1)),
                           np.array(range(-window_side, window_side + 1)), indexing='ij')

        def func_trans(variable):
            return self.cross_correlation_func_trans(variable, point, interpolated_reference, interpolated_deformed, grid, interpolationConfig)

        def func_full(variable):
            return self.cross_correlation_func_full(variable, point, interpolated_reference, interpolated_deformed, grid, interpolationConfig)

        boundTrans = [(-point[0], -point[1], -point[2]),
                      (-point[0] + dim[0] - 1, -point[1] + dim[1] - 1, -point[2] + dim[2] - 1)
                      ]
        temp_result = opt.least_squares(func_trans, np.array([0, 0, 0]), bounds=boundTrans)
        if onlyTranslationConfig:
            return temp_result.x

        bounds = [(-point[0], -point[1], -point[2],
                   -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5),
                  (-point[0] + dim[0] - 1, -point[1] + dim[1] - 1, -point[2] + dim[2] - 1,
                   0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
                  ]

        result = opt.least_squares(func_full, np.array([temp_result.x[0], temp_result.x[1], temp_result.x[2], 0, 0, 0, 0, 0, 0, 0, 0, 0]), bounds=bounds)

        # print(result)
        return result.x
