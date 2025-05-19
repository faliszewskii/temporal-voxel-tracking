import csv
import os
import random
import string

import numpy as np
from PIL import Image


root_path = f'C:\\Users\\USER\\Documents\\Repositories\\temporal-voxel-tracking\\'

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

def savePointWithGT(points, pointsGT, config, time):
    path = root_path + f'results\\transform_test_results_{randomword(6)}.csv'
    # path = f'/home/faliszewskii/Repositories/temporal-voxel-tracking/results/transform_test_results_{randomword(6)}.csv'

    with open(path, 'a') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')

        csvwriter.writerow(['Window Size', 'Only Translation', 'Interpolation Type', 'Time'])
        csvwriter.writerow([config[0], int(config[1]), config[2], time])
        csvwriter.writerow(['Algorithm'])
        csvwriter.writerow(['x', 'y', 'z'])
        for point in points:
            csvwriter.writerow([point[0], point[1], point[2]])
        csvwriter.writerow(['Ground Truth'])
        csvwriter.writerow(['x', 'y', 'z'])
        for point in pointsGT:
            csvwriter.writerow([point[0], point[1], point[2]])

def saveArray(relDir, array):
    path = root_path + relDir
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        np.save(f, array)

def loadArray(relDir):
    path = root_path + relDir
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return np.load(f)


def savePointsWithGT(points, pointsGT, pointsCount, config, times, relPath=None):
    path = root_path
    fileCode = randomword(6)
    if path is None:
        path += f'results\\transform_test_results_{fileCode}.csv'
    else:
        path += relPath
    # path = f'/home/faliszewskii/Repositories/temporal-voxel-tracking/results/transform_test_results_{randomword(6)}.csv'

    frameCount = len(points) // pointsCount
    with open(path, 'a') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        labels = ['Window Size', 'Only Translation', 'Interpolation Type', 'Time']
        for i in range(frameCount):
            labels += [f"AP_{i}_x", f"AP_{i}_y", f"AP_{i}_z"]
        for i in range(frameCount):
            labels += [f"GT_{i}_x", f"GT_{i}_y", f"GT_{i}_z"]
        csvwriter.writerow(labels)

        for i in range(pointsCount):
            row = [config[0], int(config[1]), config[2], times[i]]
            for frame in range(frameCount):
                point = points[i * frameCount + frame]
                row += [point[0], point[1], point[2]]
            for frame in range(frameCount):
                point = pointsGT[i * frameCount + frame]
                row += [point[0], point[1], point[2]]
            csvwriter.writerow(row)

    print(f"result saved to {path}")


def loadPointsWithGT(relPath):
    path = root_path + relPath

    points = []
    pointsGT = []
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Skip header row

        # Determine number of frames by counting AP_x columns
        ap_columns = [col for col in header if col.startswith('AP_') and col.endswith('_x')]
        frameCount = len(ap_columns)

        for row in csvreader:
            row_points = []
            row_pointsGT = []

            # Skip first 4 columns (config and time)
            base_idx = 4
            # Read points
            for i in range(frameCount):
                idx = base_idx + i * 3
                x, y, z = map(float, row[idx:idx + 3])
                row_points.append([x, y, z])
            # Read ground truth points
            for i in range(frameCount):
                idx = base_idx + frameCount * 3 + i * 3
                x, y, z = map(float, row[idx:idx + 3])
                row_pointsGT.append([x, y, z])

            points.extend(row_points)
            pointsGT.extend(row_pointsGT)

    return np.array(points), np.array(pointsGT)
