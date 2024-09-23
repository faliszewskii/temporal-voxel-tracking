import math
import random
import unittest

import numpy as np
import numpy.testing
import scipy.signal as signal
import scipy.optimize as opt
from PIL import Image
from matplotlib import pyplot as plt, patches
from scipy.interpolate import RegularGridInterpolator
from perlin_noise import PerlinNoise


class OptimisationTest(unittest.TestCase):
    def rosen_with_args(x, a, b):
        """The Rosenbrock function with additional arguments"""
        return sum(a * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0) + b

    def test_rosen(self):
        expected = np.array([1, 1, 1, 1, 1])
        epsilon = 7
        result = opt.shgo(OptimisationTest.rosen_with_args, [(-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10)], args=(0.5, 1.))
        # x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
        #result = opt.minimize(OptimisationTest.rosen_with_args, x0, method='nelder-mead', args=(0.5, 1.), options={'xatol': 1e-8, 'disp': True})
        numpy.testing.assert_almost_equal(result.x, expected, decimal=epsilon)


class DicTest(unittest.TestCase):

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

        right = window2/numpy.sqrt(sum2)
        if np.any(np.isnan(right)):
            right.fill(0.0)
        left = window1/numpy.sqrt(sum1)
        if np.any(np.isnan(left)):
            left.fill(0.0)

        diffs = left - right

        nlsc = (diffs ** 2).sum()

        return nlsc

    def find_max_cross_correlation_integer(self, im1, im2, coords, window_size):
        window_center = window_size // 2 + 1
        window_side = window_size - window_center

        x = coords[0]
        y = coords[1]

        dim = im1.shape
        cross_correlation = numpy.zeros((dim[0], dim[1]))

        for nx in range(window_center, dim[0] - window_center):
            for ny in range(window_center, dim[1] - window_center):
                range_x_start1 = nx - window_side
                range_x_end1 = nx + window_side
                range_y_start1 = ny - window_side
                range_y_end1 = ny + window_side

                cross_correlation[nx, ny] = self.normalised_least_squares_criterion(im1[x - window_side:x + window_side + 1, y - window_side:y + window_side + 1],
                                                    im2[range_x_start1:range_x_end1+1, range_y_start1:range_y_end1+1])

        return cross_correlation

    def bilinear_interpolation(self, x, y, points):
        '''Interpolate (x,y) from values associated with four points.

        The four points are a list of four triplets:  (x, y, value).
        The four points can be in any order.  They should form a rectangle.

            #>>> bilinear_interpolation(12, 5.5,
            ...                        [(10, 4, 100),
            ...                         (20, 4, 200),
            ...                         (10, 6, 150),
            ...                         (20, 6, 300)])
            165.0

        '''
        # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

        points = sorted(points)  # order points by x, then by y
        (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

        if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
            raise ValueError('points do not form a rectangle')
        if not x1 <= x <= x2 or not y1 <= y <= y2:
            raise ValueError('(x, y) not within the rectangle')

        return (q11 * (x2 - x) * (y2 - y) +
                q21 * (x - x1) * (y2 - y) +
                q12 * (x2 - x) * (y - y1) +
                q22 * (x - x1) * (y - y1)
                ) / ((x2 - x1) * (y2 - y1) + 0.0)

    def bilinear_image_interpolation(self, image, x, y):
        if math.floor(x) == x and math.floor(y) == y:
            return image[math.floor(x), math.floor(y)]
        if math.floor(x) == x:
            q1 = image[math.floor(x), math.floor(y)]
            q2 = image[math.floor(x), math.floor(y+1)]
            return  (math.floor(y+1) - y) * q1 + (y - math.floor(y)) * q2
        if math.floor(y) == y:
            q1 = image[math.floor(x), math.floor(y)]
            q2 = image[math.floor(x+1), math.floor(y)]
            return  (math.floor(x+1) - x) * q1 + (x - math.floor(x)) * q2


        left_upper = (math.floor(x), math.floor(y), image[math.floor(x), math.floor(y)])
        left_lower = (math.floor(x), math.floor(y+1), image[math.floor(x), math.floor(y+1)])
        right_upper = (math.floor(x+1), math.floor(y), image[math.floor(x+1), math.floor(y)])
        right_lower = (math.floor(x+1), math.floor(y+1), image[math.floor(x+1), math.floor(y+1)])

        return self.bilinear_interpolation(x, y, [left_upper, left_lower, right_upper, right_lower])

    def gkern(seld, l=5, sig=1.):
        """\
        creates gaussian kernel with side length `l` and a sigma of `sig`
        """
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
        kernel = np.outer(gauss, gauss)
        return kernel / np.sum(kernel)

    def cross_correlation_func_translation(self, variable, original_coords, interpolatedReference, interpolatedDeformed, grid):
        x, y = original_coords
        u, v = variable

        reference_window = (grid[0] + x, grid[1] + y)
        deformed_window = (grid[0] + x + u, grid[1] + y + v)

        gauss = self.gkern(reference_window[0].shape[0])
        window1 = interpolatedReference(reference_window, method="quintic")  # * gauss  # quintic
        window2 = interpolatedDeformed(deformed_window, method="quintic")  # * gauss  # quintic

        return self.normalised_least_squares_criterion(window1, window2)

    def cross_correlation_func_full(self, variable, original_coords, interpolatedReference, interpolatedDeformed, grid):
        x, y = original_coords
        u, v, dux, duy, dvx, dvy = variable

        reference_window = (x + grid[0], y + grid[1])
        deformed_window = (x + grid[0] + u + dux * grid[0] + duy * grid[1], y + grid[1] + v + dvx * grid[0] + dvy * grid[1])

        window1 = interpolatedReference(reference_window, method="quintic")  # quintic
        window2 = interpolatedDeformed(deformed_window, method="quintic")  # quintic

        return self.normalised_least_squares_criterion(window1, window2)

    # def test_compare_optimisation_to_integer(self):
    #     im1 = numpy.array(Image.open('Intact.tif'))
    #     im2 = numpy.array(Image.open('Deformed.tif'))
    #
    #     window_size = 21
    #     #
    #     x = 714
    #     y = 238
    #
    #     # x = 514
    #     # y = 538
    #
    #     # x = 50
    #     # y = 50
    #
    #     cc = self.find_max_cross_correlation_integer(im1, im2, (x, y), window_size)
    #     i, j = np.unravel_index(cc.argmax(), cc.shape)
    #
    #     # Optimisation method
    #     window_center = window_size // 2 + 1
    #     dim = im1.shape
    #
    #     def func(variable):
    #         # print(variable, self.bilinear_image_interpolation(cc, variable[0], variable[1]))
    #         return -self.cross_correlation_func(variable, (x, y), im1, im2, window_size)
    #
    #     # result = opt.dual_annealing(func, [(window_center, dim[0] - window_center-1), (window_center, dim[1] - window_center-1)])
    #     result = opt.minimize(func, np.array([x, y]), method='nelder-mead', bounds=[(window_center, dim[0] - window_center-1), (window_center, dim[1] - window_center-1)])
    #
    #     print()
    #     print(i, j, cc[i, j])
    #     print(result.x, self.bilinear_image_interpolation(cc, result.x[0], result.x[1]))

    def test_dic_translation(self):
        dim = (100, 100)
        r_0 = 20
        r_max = 20
        window_size = 23
        x = 33
        y = 55
        tr1 = (0, 0)
        tr2 = (10, 0)
        noise_amp = 0.000
        self.test_dic(dim, r_0, r_max, tr1, tr2, noise_amp, window_size, x, y)
    #
    # def test_dic_translation_noise(self):
    #     dim = (100, 100)
    #     r_0 = 20
    #     r_max = 20
    #     window_size = 15
    #     x = 40
    #     y = 40
    #     tr1 = (-4, 5)
    #     tr2 = (5, 7)
    #     noise_amp = 0.00
    #     self.test_dic(dim, r_0, r_max, tr1, tr2, noise_amp, window_size, x, y)
    #
    # def test_dic_pulse(self):
    #     dim = (100, 100)
    #     r_0 = 20
    #     r_max = 25
    #     window_size = 13
    #     x = 37
    #     y = 37
    #     tr1 = (0, 0)
    #     tr2 = (0, 0)
    #     noise_amp = 0.001
    #     self.test_dic(dim, r_0, r_max, tr1, tr2, noise_amp, window_size, x, y)


    def test_dic_translation_1(self):
        # x = list(range(40,41))
        # y = list(range(40,41))
        # window_size = [23]
        x = list(range(40, 61, 5))
        y = list(range(40, 61, 5))
        window_size = np.linspace(19, 31, 4)
        translation = (7, 7)
        epsilon = 1e-1

        for w in window_size:
            for i in x:
                for j in y:
                    u, v = self.dic_translation((100, 100), int(w), translation, i, j)
                    test_pass = abs(u - translation[0]) < epsilon and abs(v - translation[1]) < epsilon
                    print(f"x:{i}, y:{j}, w:{int(w)}, result:{test_pass}, error:{(abs(u - translation[0]) + abs(v - translation[1])/2)}")
    def dic_translation(self, dim, window_size, translation, x, y):
        im1 = np.zeros(dim, dtype=np.float32)
        im2 = np.zeros(dim, dtype=np.float32)
        center = np.array([d / 2 for d in dim])

        def func(x, y, f, translation):
            r = 150
            radius = np.sqrt((x - center[0] - translation[0]) ** 2 + (y - center[1] - translation[1]) ** 2)
            mask = radius > r
            values = np.where(mask, 0, radius / r)
            return values

        start_x = 0
        end_x = dim[0]
        start_y = 0
        end_y = dim[1]
        x_range = np.arange(start_x, end_x)
        y_range = np.arange(start_y, end_y)
        X, Y = np.meshgrid(x_range, y_range, indexing='ij')
        im1[start_x:end_x, start_y:end_y] = func(X, Y, 0, (0, 0))
        im2[start_x:end_x, start_y:end_y] = func(X, Y, 5, translation)

        window_center = window_size // 2 + 1
        window_side = window_size - window_center

        result = self.find_correlated_point(im1, im2, np.array([x, y]), window_size)
        u, v, dux, duy, dvx, dvy = result

        return u, v

    def test_dic(self, dim, r_0, r_max, translation1, translation2, noise_amp, window_size, x, y, verbose=False):

        frame_count = 10
        def r_func(f): return (r_0 + r_max) / 2 - (r_max - r_0) / 2 * math.cos(f / frame_count * (2 * math.pi))

        im1 = np.zeros(dim, dtype=np.float32)
        im2 = np.zeros(dim, dtype=np.float32)
        center = np.array([d / 2 for d in dim])

        def func(x, y, f, translation):
            radius = np.sqrt((x - center[0] - translation[0]) ** 2 + (y - center[1] - translation[1]) ** 2)
            mask = radius > r_func(f)
            values = np.where(mask, 0, radius / r_func(f))
            noise = np.array([[random.random() * noise_amp for i in range(dim[0])] for j in range(dim[1])])
            return values + noise

        # def func(x, y, f, translation):
        #     noise = PerlinNoise(octaves=10, seed=1)
        #     xpix, ypix = x.shape
        #     return [[ noise([(x[i,j]- translation[0])/xpix, (y[i,j] - translation[1])/ypix])  for j in range(xpix)] for i in range(ypix)]

        start_x = 0
        end_x = dim[0]
        start_y = 0
        end_y = dim[1]
        x_range = np.arange(start_x, end_x)
        y_range = np.arange(start_y, end_y)
        X, Y = np.meshgrid(x_range, y_range, indexing='ij')
        im1[start_x:end_x, start_y:end_y] = func(X, Y, 0, translation1)
        im2[start_x:end_x, start_y:end_y] = func(X, Y, 5, translation2)

        # cc = self.find_max_cross_correlation_integer(im1, im2, (x, y), window_size)
        # i, j = np.unravel_index(cc.argmax(), cc.shape)

        # Optimisation method
        window_center = window_size // 2 + 1
        dim = im1.shape
        window_side = window_size - window_center

        result = self.find_correlated_point(im1, im2, np.array([x, y]), window_size)
        #
        # def func(variable):
        #     print(variable, self.bilinear_image_interpolation(cc, variable[0], variable[1]))
        #     return self.cross_correlation_func(variable, (x, y), im1, im2, window_size)
        #
        # result = opt.least_squares(func, np.array([x, y]), bounds=[(window_center, window_center), (dim[0] - window_center-1, dim[1] - window_center-1)])
        # result = opt.shgo(func, [(window_center, dim[0] - window_center-1), (window_center, dim[1] - window_center-1)])
        # result = opt.dual_annealing(func, [(window_center, dim[0] - window_center-1), (window_center, dim[1] - window_center-1)])
        # result = opt.minimize(func, np.array([x, y]), method='nelder-mead', bounds=[(window_center, dim[0] - window_center-1), (window_center, dim[1] - window_center-1)])
        # result = opt.minimize(func, np.array([x, y]), method='BFGS', bounds=[(window_center, dim[0] - window_center-1), (window_center, dim[1] - window_center-1)])

        # print()
        # print(i, j, cc[i, j])
        # print(result.x, self.bilinear_image_interpolation(cc, result.x[0], result.x[1]))

        u, v, dux, duy, dvx, dvy = result
        grid = np.meshgrid(np.array(range(-window_side, window_side + 1)),
                                np.array(range(-window_side, window_side + 1)), indexing='ij')
        reference_window = (x + grid[0], y + grid[1])
        deformed_window = (x + grid[0] + u + dux * grid[0] + duy * grid[1], y + grid[1] + v + dvx * grid[0] + dvy * grid[1])

        # fig, ax = plt.subplots()
        # ax.imshow(cc.T, cmap='gray', origin='lower')
        # plt.show()

        fig, ax = plt.subplots()
        ax.imshow(im1.T, cmap='gray', interpolation='nearest', origin='lower')
        ax.scatter(x, y, s=4, c='red', marker='x')
        ax.scatter(reference_window[0], reference_window[1], s=1, c='red', marker='o')
        # rect = patches.Rectangle((x-window_size//2-1,y-window_size//2-1), window_size, window_size, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        plt.show()
        fig2, ax2 = plt.subplots()
        ax2.imshow(im2.T, cmap='gray', interpolation='nearest', origin='lower')
        ax2.scatter(x + u, y + v, s=4, c='red', marker='x')
        ax2.scatter(deformed_window[0], deformed_window[1], s=1, c='red', marker='o')
        # rect = patches.Rectangle((result[1]-window_size//2-1,result[0]-window_size//2-1), window_size, window_size, linewidth=1, edgecolor='r', facecolor='none')
        # ax2.add_patch(rect)
        plt.show()

    def find_correlated_point(self, reference_volume, deformed_volume, point, window_size):
        '''
        reference_volume, deformed_volume - 2D numpy array
        point - tuple (x, y)
        '''

        dim = reference_volume.shape

        # window_size = 15
        window_center = window_size // 2 + 1
        window_side = window_size - window_center

        range_x = np.array(range(0, deformed_volume.shape[0]))
        range_y = np.array(range(0, deformed_volume.shape[1]))
        interpolatedReference = RegularGridInterpolator((range_x, range_y), reference_volume, fill_value=0.0, bounds_error=False)
        interpolatedDeformed = RegularGridInterpolator((range_x, range_y), deformed_volume, fill_value=0.0, bounds_error=False)

        grid = np.meshgrid(np.array(range(-window_side, window_side + 1)),
                                np.array(range(-window_side, window_side + 1)), indexing='ij')

        def func_translation(variable):
            return self.cross_correlation_func_translation(variable, point, interpolatedReference, interpolatedDeformed, grid)

        def func_full(variable):
            return self.cross_correlation_func_full(variable, point, interpolatedReference, interpolatedDeformed, grid)

        # temp_result = opt.minimize(func_translation, np.array([0, 0]), method='nelder-mead',bounds=[(-point[0] + window_center, -point[1] + window_center),
        #                                                 (-point[0] + dim[0] - window_center - 1,
        #                                                  -point[1] + dim[1] - window_center - 1)])

        temp_result = opt.least_squares(func_translation, np.array([0, 0]), bounds=[(-point[0], -point[1]),
                                                        (-point[0] + dim[0] - 1,
                                                         -point[1] + dim[1] - 1)],
                                        jac='3-point', method='trf')
        # print(temp_result)

        # result = opt.least_squares(func_full, np.array([temp_result.x[0], temp_result.x[1], 0, 0, 0, 0]),
        #                            bounds=[(-point[0] + window_center, -point[1] + window_center, -0.8, -0.8, -0.8, -0.8),
        #                                                 (-point[0] + dim[0] - window_center - 1,
        #                                                  -point[1] + dim[1] - window_center - 1, 1, 1, 1, 1)])
        # print(result)

        return (temp_result.x[0], temp_result.x[1], 0, 0, 0, 0)

    def test_interpolation(self):
        img = np.array([[5, 7, 8, 7, 3, 5],
                        [5, 4, 7, 8, 2, 1],
                        [9, 10, 11, 12, 15, 9],
                        [13, 14, 5, 16, 13, 12],
                        [9, 8, 11, 20, 14, 7],
                        [3, 6, 1, 3, 0, 1]])

        interpolated = np.zeros((img.shape[0]*50, img.shape[1]*50))

        x = np.array(range(0, img.shape[0]))
        y = np.array(range(0, img.shape[1]))
        interp = RegularGridInterpolator((x, y), img)

        for i in range((img.shape[0]-1)*50):
            for j in range((img.shape[1]-1)*50):
                # interpolated[i, j] = self.bilinear_image_interpolation(img, i/50.0, j/50.0)
                interpolated[i, j] = interp([[i/50.0, j/50.0]], method='pchip')[0]
        plt.imshow(interpolated, cmap='gray')
        plt.show()
        # def func(variable):
        #     print(variable)
        #     return self.bilinear_image_interpolation(img, variable[0], variable[1])
        #
        # result = opt.dual_annealing(func, [(0, 3), (0, 3)])
        # print(result)
        # # print(result.x, self.bilinear_image_interpolation(interpolated, result.x[0], result.x[1]))


if __name__ == '__main__':
    unittest.main()
