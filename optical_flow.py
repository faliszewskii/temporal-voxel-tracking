import numpy as np
import scipy


def compute_optical_flow(current_frame, next_frame):  # Takes a 3d numpy array
    dim = current_frame.shape

    i_z, i_y, i_x = np.gradient(current_frame)
    # i_x = scipy.signal.convolve(current_frame, sobel_x(), mode='same')
    # i_y = scipy.signal.convolve(current_frame, sobel_y(), mode='same')
    # i_z = scipy.signal.convolve(current_frame, sobel_z(), mode='same')

    i_t = np.subtract(next_frame, current_frame)
    # (i_x, i_y, i_z) * (v_x, v_y,v_z) = -i_t

    mean_size = 5
    kern = np.ones((mean_size, mean_size, mean_size)) / mean_size**3
    alpha = 10
    iterations = 100
    v_x = np.zeros((dim[0], dim[1], dim[2]))
    v_y = np.zeros((dim[0], dim[1], dim[2]))
    v_z = np.zeros((dim[0], dim[1], dim[2]))
    m_v_x = np.zeros((dim[0], dim[1], dim[2]))
    m_v_y = np.zeros((dim[0], dim[1], dim[2]))
    m_v_z = np.zeros((dim[0], dim[1], dim[2]))
    for it in range(iterations):
        factor = (i_x*m_v_x + i_y*m_v_y + i_z*m_v_z + i_t) / (alpha*alpha + i_x*i_x + i_y*i_y + i_z*i_z)
        v_x = m_v_x - i_x * factor
        v_y = m_v_y - i_y * factor
        v_z = m_v_z - i_z * factor
        m_v_x = scipy.signal.convolve(v_x, kern, mode='same')
        m_v_y = scipy.signal.convolve(v_y, kern, mode='same')
        m_v_z = scipy.signal.convolve(v_z, kern, mode='same')
        if iterations < 10 or it % int(iterations/10) == 0:
            print(f'{it},')
    res = np.stack((v_z, v_y, v_x), axis=-1)
    return res


# def sobel_x():
#     return np.array([
#         [[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]],
#         [[ 2,  0, -2], [ 4,  0, -4], [ 2,  0, -2]],
#         [[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]]
#     ])
#
#
# def sobel_y():
#     return np.array([
#         [[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]],
#         [[ 2,  4,  2], [ 0,  0,  0], [-2, -4, -2]],
#         [[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]]
#     ])
#
#
# def sobel_z():
#     return np.array([
#         [[ 1,  2,  1], [ 2,  4,  2], [ 1,  2,  1]],
#         [[ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0]],
#         [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
#     ])
