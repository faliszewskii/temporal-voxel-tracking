import time
import digital_volume_correlation as dvc
from dvc.spline_interpolation import *

class VoxelTracker:
    def __init__(self):
        self.dvc = dvc.DigitalVolumeCorrelation()

    def track(self, frames, spline_interpolators, begin_frame, start, windowSizeConfig, onlyTranslationConfig, interpolationConfig):
        frame_count = frames.shape[0]
        points = np.zeros((frame_count, 3))
        correlations = []
        reference_t = begin_frame

        # Current frame
        points[begin_frame] = np.array(start)
        current_t = begin_frame + 1

        threshold = 0.1  # Least Squares

        usePreviousFrame = False

        if usePreviousFrame:
            for i in range(begin_frame+1, frame_count - begin_frame):
                t = (i + begin_frame)
                start = time.time()
                u, v, w, c_ls = self.dvc.find_correlated_point(frames[t-1], frames[t], spline_interpolators[t-1], spline_interpolators[t], points[t-1], windowSizeConfig, onlyTranslationConfig, interpolationConfig)
                end = time.time()
                print(f"{t-1} -> {t}: Translation found: {u:2f}, {v:2f}, {w:2f} with correlation: {c_ls} in time: {end-start}.")
                correlations.append(c_ls)
                points[t] = points[t-1] + np.array([u, v, w])
        else:
            while current_t < frame_count - begin_frame:
                u, v, w, c_ls = self.dvc.find_correlated_point(frames[reference_t], frames[current_t], spline_interpolators[reference_t], spline_interpolators[current_t], points[reference_t], windowSizeConfig, onlyTranslationConfig, interpolationConfig)
                print(f"{reference_t} -> {current_t}: Translation found: {u:2f}, {v:2f}, {w:2f} with correlation: {c_ls}.")
                if c_ls > threshold and reference_t != current_t - 1:
                    # Bad correlation and we can change the reference
                    print(f"{reference_t} -> {current_t}: Trying again.")
                    reference_t = current_t - 1
                else:
                    print(f"{reference_t} -> {current_t}: Saving point.")
                    points[current_t] = points[reference_t] + np.array([u, v, w])
                    correlations.append(c_ls)
                    current_t += 1

        return points, correlations
