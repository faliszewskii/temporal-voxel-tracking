import numpy as np
import digital_volume_correlation as dvc


class VoxelTracker:
    def __init__(self):
        self.dvc = dvc.DigitalVolumeCorrelation()

    def track(self, frames, begin_frame, start, windowSizeConfig, onlyTranslationConfig, interpolationConfig):
        frame_count = frames.shape[0]
        points = np.zeros((frame_count, 3))
        correlations = []
        reference_t = begin_frame

        # Current frame
        points[begin_frame] = np.array(start)
        current_t = begin_frame + 1

        threshold = 0.4  # Least Squares

        usePreviousFrame = True

        if usePreviousFrame:
            for i in range(begin_frame+1, frame_count - begin_frame):
                t = (i + begin_frame)
                u, v, w, c_ls = self.dvc.find_correlated_point(frames[t-1], frames[t], points[t-1], windowSizeConfig, onlyTranslationConfig, interpolationConfig)
                correlations.append(c_ls)
                points[t] = points[t-1] + np.array([u, v, w])
        else:
            while current_t < frame_count - begin_frame:
                u, v, w, c_ls = self.dvc.find_correlated_point(frames[reference_t], frames[current_t], points[reference_t], windowSizeConfig, onlyTranslationConfig, interpolationConfig)
                print(f"{reference_t} -> {current_t}: Translation found: {u}, {v}, {w} with correlation: {c_ls}.")
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
