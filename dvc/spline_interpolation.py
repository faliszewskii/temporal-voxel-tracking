import numpy as np
from scipy.fft import fft, ifft


class Spline5Interpolator3d:

    def __init__(self, data, padding=3):
        self.padding = padding
        self.coefficients = self.calculate_spline5_coefficients(data)
        self.basis_matrix_dictionary = dict()
        self.dim = data.shape
        self.exp_to_bs5 = np.array([
            [1/120,  13/60, 11/20, 13/60, 1/120, 0],
            [-1/24,  -5/12, 0,     5/12,  1/24,  0],
            [1/12,   1/6,   -1/2,  1/6,   1/12,  0],
            [-1/12,  1/6,   0,     -1/6,  1/12,  0],
            [1/24,   -1/6,  1/4,   -1/6,  1/24,  0],
            [-1/120, 1/24,  -1/12, 1/12,  -1/24, 1/120]
        ])  # (6, 6)
        offset_range = np.arange(-2, 4)
        ox, oy, oz = np.meshgrid(offset_range, offset_range, offset_range, indexing='ij')
        self.ox = ox.ravel()
        self.oy = oy.ravel()
        self.oz = oz.ravel()

    def calculate_spline5_coefficients(self, data):
        eps = 1e-5
        kernel = [1/120, 13/60, 11/20, 13/60, 1/120, 0]
        result = np.pad(data, self.padding, mode='edge')
        dims = result.shape

        for d in range(len(dims)):
            dim = dims[d]
            padded_kernel = np.pad(kernel, (0, dim-6), 'constant', constant_values=0)
            padded_kernel = np.roll(padded_kernel, -2)

            freq_kernel = fft(padded_kernel)
            freq_kernel[np.abs(freq_kernel) < eps] = eps
            freq_data = fft(result, axis=d)

            reshape_dims = [1] * len(dims)
            reshape_dims[d] = freq_kernel.shape[0]
            freq_kernel = freq_kernel.reshape(reshape_dims)

            result = ifft(freq_data / freq_kernel, axis=d)

        return np.real(result)

    def interpolate(self, grid):
        coords = np.stack([g.flatten() for g in grid], axis=1)
        fxArr = np.floor(coords).astype(int)
        dxArr = coords - fxArr
        indices = np.array([self.encode_xyz(fx[0], fx[1], fx[2], self.dim[1], self.dim[2]) for fx in fxArr])

        is_cached = np.array([i in self.basis_matrix_dictionary for i in indices])
        uncached_fxs = fxArr[~is_cached]
        uncached_indices = indices[~is_cached]

        if len(uncached_fxs) > 0:
            print(f'Uncached indices: {len(uncached_indices)}')
            print(f'Dictionary size: {len(self.basis_matrix_dictionary)}')

            N = uncached_fxs.shape[0]
            ix = uncached_fxs[:, 0][:, None] + self.ox[None, :] + self.padding  # (N, 216)
            iy = uncached_fxs[:, 1][:, None] + self.oy[None, :] + self.padding
            iz = uncached_fxs[:, 2][:, None] + self.oz[None, :] + self.padding

            c = self.coefficients[ix, iy, iz].reshape(N, 6, 6, 6)

            c = np.tensordot(c, self.exp_to_bs5, axes=([3], [1]))  # (N, 6, 6, 6)
            c = np.tensordot(c, self.exp_to_bs5, axes=([2], [1]))  # (N, 6, 6, 6)
            c = np.tensordot(c, self.exp_to_bs5, axes=([1], [1]))  # (N, 6, 6, 6)

            for idx, w in zip(uncached_indices, c):
                self.basis_matrix_dictionary[idx] = w

        dx_powers = dxArr[..., None] ** np.arange(6)  # shape: (N, 3, 6)
        vx = dx_powers[:, 0, :]
        vy = dx_powers[:, 1, :]
        vz = dx_powers[:, 2, :]

        v = vx[:, None, None, :] * vy[:, None, :, None] * vz[:, :, None, None]  # (N,6,6,6)

        weights = np.array([self.basis_matrix_dictionary[i] for i in indices])  # shape: (N, 6, 6, 6)
        result = np.sum(v*weights, axis=(1, 2, 3))  # (N,)

        return result.reshape(grid[0].shape)

    @staticmethod
    def encode_xyz(x, y, z, max_y=256, max_z=256):
        return x * (max_y * max_z) + y * max_z + z
