import csv
import os
import random
import string

import numpy as np
from PIL import Image


def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

def savePointWithGT(points, pointsGT, config, time):
    path = f'C:\\Users\\USER\\Documents\\Repositories\\temporal-voxel-tracking\\results\\transform_test_results_{randomword(6)}.csv'
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
    path = f'C:\\Users\\USER\\Documents\\Repositories\\temporal-voxel-tracking\\{relDir}'

    with open(path, 'wb') as f:
        np.save(f, array)

def loadArray(relDir):
    path = f'C:\\Users\\USER\\Documents\\Repositories\\temporal-voxel-tracking\\{relDir}'
    with open(path, 'rb') as f:
        return np.load(f)
