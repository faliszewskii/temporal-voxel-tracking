import numpy as np
import scipy


def compute_optical_flow(current_frame, next_frame):  # Takes a 3d numpy array
    dim = current_frame.shape

    i_x = scipy.signal.convolve(current_frame, sobel_x(), mode='same')
    i_y = scipy.signal.convolve(current_frame, sobel_y(), mode='same')
    i_z = scipy.signal.convolve(current_frame, sobel_z(), mode='same')

    i_t = np.subtract(next_frame, current_frame)
    # (i_x, i_y, i_z) * (v_x, v_y,v_z) = -i_t

    kern = np.ones((5, 5, 5)) / (5 * 5 * 5)
    alpha = 1
    iterations = 10
    v_x = np.zeros((current_frame.shape[0], current_frame.shape[1], current_frame.shape[2]))
    v_y = np.zeros((current_frame.shape[0], current_frame.shape[1], current_frame.shape[2]))
    v_z = np.zeros((current_frame.shape[0], current_frame.shape[1], current_frame.shape[2]))
    m_v_x = np.zeros((current_frame.shape[0], current_frame.shape[1], current_frame.shape[2]))
    m_v_y = np.zeros((current_frame.shape[0], current_frame.shape[1], current_frame.shape[2]))
    m_v_z = np.zeros((current_frame.shape[0], current_frame.shape[1], current_frame.shape[2]))
    for it in range(iterations):
        print(f'Iteration {it}')
        factor = (i_x*m_v_x + i_y*m_v_y + i_z*m_v_z + i_t) / (alpha*alpha + i_x*i_x + i_y*i_y + i_z*i_z)
        v_x = m_v_x - i_x * factor
        v_y = m_v_y - i_y * factor
        v_z = m_v_z - i_z * factor
        m_v_x = scipy.signal.convolve(v_x, kern, mode='same')
        m_v_y = scipy.signal.convolve(v_y, kern, mode='same')
        m_v_z = scipy.signal.convolve(v_z, kern, mode='same')
        # for x in range(dim[0]):
        #     for y in range(dim[1]):
        #         print(f'Iteration {it}: {x}, {y}')
        #         for z in range(dim[2]):
                    # m_v_x[x][y][z] = 0
                    # m_v_y[x][y][z] = 0
                    # m_v_z[x][y][z] = 0
                    # n = 0
                    # for i in range(-2, 3):
                    #     for j in range(-2, 3):
                    #         for k in range(-2, 3):
                    #             if x + i < 0 or y + j < 0 or z + k < 0 or x + i >= dim[0] or y + j >= dim[1] or z + k >= dim[2]:
                    #                 continue
                    #             n += 1
                    #             m_v_x[x][y][z] += v_x[x + i][y + j][z + k]
                    #             m_v_y[x][y][z] += v_y[x + i][y + j][z + k]
                    #             m_v_z[x][y][z] += v_z[x + i][y + j][z + k]
                    # m_v_x[x][y][z] /= n
                    # m_v_y[x][y][z] /= n
                    # m_v_z[x][y][z] /= n

    return np.stack((v_x, v_y, v_z), axis=-1)


def sobel_x():
    return np.array([
        [[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]],
        [[ 2,  0, -2], [ 4,  0, -4], [ 2,  0, -2]],
        [[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]]
    ])


def sobel_y():
    return np.array([
        [[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]],
        [[ 2,  4,  2], [ 0,  0,  0], [-2, -4, -2]],
        [[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]]
    ])


def sobel_z():
    return np.array([
        [[ 1,  2,  1], [ 2,  4,  2], [ 1,  2,  1]],
        [[ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0]],
        [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
    ])
