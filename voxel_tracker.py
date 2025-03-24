import numpy as np
import digital_volume_correlation as dvc


class VoxelTracker:
    def __init__(self):
        self.dvc = dvc.DigitalVolumeCorrelation()

    def track(self, frames, current_frame, start, windowSizeConfig, onlyTranslationConfig, useKTConfig, interpolationConfig):
        frame_count = frames.shape[0]
        points = np.zeros((frame_count, 3))

        # Current frame
        points[current_frame] = np.array(start)
        for i in range(frame_count - current_frame - 1):
            t = (i + current_frame)

            if useKTConfig:
                h = 1.0
                k1 = self.dvc.find_correlated_point(frames, t, points[t], windowSizeConfig, onlyTranslationConfig, interpolationConfig)[0:3]
                k2 = self.dvc.find_correlated_point(frames, t + h/2, points[t] + h*k1/2, windowSizeConfig, onlyTranslationConfig, interpolationConfig)[0:3]
                k3 = self.dvc.find_correlated_point(frames, t + h/2, points[t] + h * k2 / 2, windowSizeConfig, onlyTranslationConfig, interpolationConfig)[0:3]
                k4 = self.dvc.find_correlated_point(frames, t + h, points[t] + h * k3, windowSizeConfig, onlyTranslationConfig, interpolationConfig)[0:3]
                points[t + 1] = points[t] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                u, v, w = self.dvc.find_correlated_point(frames, t, points[t], windowSizeConfig, onlyTranslationConfig, interpolationConfig)[0:3]  # Naive Euler
                points[t + 1] = points[t] + np.array([u, v, w])

            # u, v, w = self.dvc.find_correlated_point(frames[t], frames[t+1], points[t], windowSizeConfig, onlyTranslationConfig)[0:3]  # Naive Euler
            # points[t + 1] = points[t] + np.array([u, v, w])

        return points
