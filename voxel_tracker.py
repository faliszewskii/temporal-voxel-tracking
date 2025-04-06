import numpy as np
import digital_volume_correlation as dvc


class VoxelTracker:
    def __init__(self):
        self.dvc = dvc.DigitalVolumeCorrelation()

    def track(self, frames, current_frame, start, windowSizeConfig, onlyTranslationConfig, interpolationConfig):
        frame_count = frames.shape[0]
        points = np.zeros((frame_count, 3))

        # Current frame
        points[current_frame] = np.array(start)
        for i in range(frame_count - current_frame - 1):
            t = (i + current_frame)

            u, v, w = self.dvc.find_correlated_point(frames[t], frames[t+1], points[t], windowSizeConfig, onlyTranslationConfig, interpolationConfig)[0:3]  # Naive Euler
            points[t + 1] = points[t] + np.array([u, v, w])

        return points
