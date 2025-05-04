import math

import ctypes
import numpy as np
import os


class FrameGenerator:
    def __init__(self):
        pass

    def generateFrames(self, startFrame, totalFrameCount, transformType, isCyclical):
        framesToGenerate = totalFrameCount - 1
        if isCyclical:
            framesToGenerate -= 1

        # lib_path = os.path.abspath('./libframe_generator.dll')  # TODO Returns 'C:\Users\USER\libframe_generator.dll'
        lib_path = f'C:\\Users\\USER\\Documents\\Repositories\\temporal-voxel-tracking\\libframe_generator.dll'
        lib = ctypes.CDLL(lib_path)

        lib.generateFrames.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # Input frame
            ctypes.POINTER(ctypes.c_float),  # Output frames
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # Dimensions
            ctypes.c_int,  # Frame count
            ctypes.c_int,  # Transform function enum
            ctypes.c_bool,  # Is cyclical
        ]
        lib.generateFrames.restype = None

        dim = startFrame.shape
        transformedFrame = np.empty((framesToGenerate, dim[0], dim[1], dim[2]), dtype=np.float32, order='C')

        start_ptr = startFrame.reshape(dim, order='C').flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        transformed_ptr = transformedFrame.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        lib.generateFrames(start_ptr, transformed_ptr, dim[0], dim[1], dim[2], totalFrameCount, transformType, isCyclical)

        transformedFrame = np.ctypeslib.as_array(transformed_ptr, shape=(framesToGenerate * dim[0] * dim[1] * dim[2],))
        transformedFrame = transformedFrame.reshape((framesToGenerate, dim[0], dim[1], dim[2]), order='C').astype(np.float32)

        outputFrames = [transformedFrame[i] for i in range(transformedFrame.shape[0])]

        outputFrames = [startFrame] + outputFrames
        if isCyclical:
            outputFrames = outputFrames + [startFrame]

        return outputFrames