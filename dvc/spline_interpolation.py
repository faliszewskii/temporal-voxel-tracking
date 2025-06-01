
import numpy as np
from scipy.fft import fft, ifft

def calculate_spline5_coefficients(data, padding=3):
    eps = 1e-5
    kernel = [1/120, 13/60, 11/20, 13/60, 1/120, 0]
    result = np.pad(data, padding, mode='edge')
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


def interpolate_spline5(grid, coeffs, padding=3, batch_size=100_000):
    exp_to_bs5 = np.array([
        [1/120,  13/60, 11/20, 13/60, 1/120, 0],
        [-1/24,  -5/12, 0,     5/12,  1/24,  0],
        [1/12,   1/6,   -1/2,  1/6,   1/12,  0],
        [-1/12,  1/6,   0,     -1/6,  1/12,  0],
        [1/24,   -1/6,  1/4,   -1/6,  1/24,  0],
        [-1/120, 1/24,  -1/12, 1/12,  -1/24, 1/120]
    ])

    coords = np.stack([g.flatten() for g in grid], axis=0)  # (3, N)
    N = coords.shape[1]
    results = []

    offset_range = np.arange(-2, 4)
    ox, oy, oz = np.meshgrid(offset_range, offset_range, offset_range, indexing='ij')
    ox = ox.ravel()
    oy = oy.ravel()
    oz = oz.ravel()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        coords_batch = coords[:, start:end]  # (3, B)
        B = coords_batch.shape[1]

        fx = np.floor(coords_batch).astype(int)
        dx = coords_batch - fx

        # Build basis weights
        v = np.stack([np.ones(B), dx[0], dx[0]**2, dx[0]**3, dx[0]**4, dx[0]**5], axis=1)
        wx = v @ exp_to_bs5  # (B, 6)

        v = np.stack([np.ones(B), dx[1], dx[1]**2, dx[1]**3, dx[1]**4, dx[1]**5], axis=1)
        wy = v @ exp_to_bs5

        v = np.stack([np.ones(B), dx[2], dx[2]**2, dx[2]**3, dx[2]**4, dx[2]**5], axis=1)
        wz = v @ exp_to_bs5

        ix = fx[0][:, None] + ox[None, :] + padding  # (B, 216)
        iy = fx[1][:, None] + oy[None, :] + padding
        iz = fx[2][:, None] + oz[None, :] + padding

        c = coeffs[ix, iy, iz]  # (B, 216)

        # Compute weight products with broadcasting
        wx_ = wx[:, :, None, None]
        wy_ = wy[:, None, :, None]
        wz_ = wz[:, None, None, :]

        weights = (wx_ * wy_ * wz_).reshape(B, 216)  # (B, 216)
        result_batch = np.sum(weights * c, axis=1)  # (B,)
        results.append(result_batch)

    result = np.concatenate(results, axis=0)
    return result.reshape(grid[0].shape)

