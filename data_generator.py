import math

import slicer
import vtk
import numpy as np
from vtk.util import numpy_support
from numpy import random

from generated_data import GeneratedData


class DataGenerator:

    def __init__(self):
        pass

    def generate_data(self, frame_count, data_func, tracker):
        sequence_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode")
        sequence_node.SetName("Generated Data Sequence")
        sequence_node.SetIndexName("frames")
        sequence_node.SetIndexUnit("frame")

        seq_browser = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode")
        seq_browser.SetName("Generated Data Browser")
        seq_browser.SetAndObserveMasterSequenceNodeID(sequence_node.GetID())
        seq_browser.SetSaveChanges(sequence_node, True)

        dimensions = (64, 64, 64)

        for i in range(frame_count):
            data_vtk = numpy_support.numpy_to_vtk(num_array=data_func(dimensions, i).ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

            image_data = vtk.vtkImageData()
            image_data.SetDimensions(dimensions)
            image_data.AllocateScalars(vtk.VTK_FLOAT, 1)
            image_data.GetPointData().SetScalars(data_vtk)

            volume_node = slicer.vtkMRMLScalarVolumeNode()
            volume_node.SetOrigin(0, 0, 0)
            volume_node.SetSpacing(1, 1, 1)
            volume_node.SetIJKToRASDirectionMatrix(vtk.vtkMatrix4x4())
            volume_node.SetAndObserveImageData(image_data)

            sequence_node.SetDataNodeAtValue(volume_node, str(i))
        return GeneratedData(sequence_node, seq_browser, tracker)

    @staticmethod
    def linear_cube_data(dimensions, i, delta):
        cubeArray = np.zeros(dimensions, dtype=np.float32)

        # Define the size of the cube
        cube_size = dimensions[0] / 2
        half_cube_size = cube_size // 2

        # Calculate the center of the volume
        center = np.array([dim / 2 for dim in dimensions])
        center += delta[i]

        start_x = int(max(center[0] - half_cube_size, 0))
        end_x = int(min(center[0] + half_cube_size, dimensions[0]))
        start_y = int(max(center[1] - half_cube_size, 0))
        end_y = int(min(center[1] + half_cube_size, dimensions[1]))
        start_z = int(max(center[2] - half_cube_size, 0))
        end_z = int(min(center[2] + half_cube_size, dimensions[2]))

        # Fill the array with the cube values
        def func(x, y, z):
            return (x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2
        x_range = np.arange(start_x, end_x)
        y_range = np.arange(start_y, end_y)
        z_range = np.arange(start_z, end_z)
        X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        cubeArray[start_x:end_x, start_y:end_y, start_z:end_z] = func(X, Y, Z)

        return cubeArray

    def generate_static_cube(self):
        print("Generating static cube...")
        frame_count = 10
        def f(x): return np.array([0.0, 0.0, 0.0])
        def tracker(p0, t_p0, t) :
            return p0
        return self.generate_data(frame_count, lambda dimensions, i: self.linear_cube_data(dimensions, i, [f(x) for x in np.arange(frame_count)]), tracker)

    def generate_slow_cube(self):
        print("Generating slow cube...")
        frame_count = 10
        def f(x): return x * np.array([0.0, 0.0, 1.0])
        def tracker(p0, t_p0, t) :
            return p0 + f(t - t_p0)
        return self.generate_data(frame_count, lambda dimensions, i: self.linear_cube_data(dimensions, i, [f(x) for x in np.arange(frame_count)]), tracker)

    def generate_faster_cube(self):
        print("Generating faster cube...")
        frame_count = 10
        def f(x): return x * np.array([2.0, 0.0, 0.0])
        def tracker(p0, t_p0, t) :
            return p0 + f(t - t_p0)
        return self.generate_data(frame_count, lambda dimensions, i: self.linear_cube_data(dimensions, i, [f(x) for x in np.arange(frame_count)]), tracker)

    def generate_random_cube(self):
        print("Generating random cube...")
        frame_count = 10
        current_pos = [0.0, 0.0, 0.0]
        def f(x): return 10 * np.array([random.random(), random.random(), random.random()]) + current_pos
        def tracker(p0, t_p0, t): return None
        return self.generate_data(frame_count, lambda dimensions, i: self.linear_cube_data(dimensions, i, [f(x) for x in np.arange(frame_count)]), tracker)

    @staticmethod
    def cylinder_func(dim, frame, rFunc):
        data_array = np.zeros(dim, dtype=np.float32)
        center = np.array([d / 2 for d in dim])

        def func(x, y, z):
            radius = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            mask = radius > rFunc(frame)
            values = np.where(mask, 0, radius / rFunc(frame))
            return values
        start_x = 0
        end_x = dim[0]
        start_y = 0
        end_y = dim[1]
        start_z = 0
        end_z = dim[2]
        x_range = np.arange(start_x, end_x)
        y_range = np.arange(start_y, end_y)
        z_range = np.arange(start_z, end_z)
        X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        data_array[start_x:end_x, start_y:end_y, start_z:end_z] = func(X, Y, Z)
        return data_array

    def generate_pulsating_cylinder(self):
        print("Generating pulsating cylinder...")
        dimensions = (64, 64, 64)
        center = np.array([d / 2 for d in dimensions])
        frame_count = 10
        r_0 = 20
        r_max = 23
        def r_func(f): return (r_0 + r_max) / 2 - (r_max - r_0) / 2 * math.cos(f / frame_count * (2 * math.pi))
        def cylinder(dim, frame): return self.cylinder_func(dim, frame, r_func)

        def tracker(p0, t_p0, t):
            c = np.array([center[0], center[1], p0[2]])
            return c + (p0 - c) * r_func(t) / r_func(t_p0)
        return self.generate_data(frame_count, cylinder, tracker)

    # @staticmethod
    # def bezier_func(dim, frame, rFunc):
    #     data_array = np.zeros(dim, dtype=np.float32)
    #     center = np.array([d / 2 for d in dim])
    #
    #     def func(x, y, z):
    #         radius = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    #         mask = radius > rFunc(frame)
    #         values = np.where(mask, 0, radius / rFunc(frame))
    #         return values
    #
    #     start_x = 0
    #     end_x = dim[0]
    #     start_y = 0
    #     end_y = dim[1]
    #     start_z = 0
    #     end_z = dim[2]
    #     x_range = np.arange(start_x, end_x)
    #     y_range = np.arange(start_y, end_y)
    #     z_range = np.arange(start_z, end_z)
    #     X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    #     data_array[start_x:end_x, start_y:end_y, start_z:end_z] = func(X, Y, Z)
    #     return data_array
    #
    # def generate_pulsating_cylinder(self):
    #     print("Generating pulsating bezier...")
    #     dimensions = (64, 64, 64)
    #     frame_count = 10
    #     r_0 = 15
    #     r_max = 17
    #     xs = np.arange(4) * dimensions[0] / 4
    #     ys = random.rand(4) * dimensions[1]
    #     zs = random.rand(4) * dimensions[2]
    #     bezier_points = np.vstack((xs, ys, zs)).T
    #
    #     def r_func(f): return (r_0 + r_max) / 2 - (r_max - r_0) / 2 * math.cos(f / frame_count * (2 * math.pi))
    #     def bezier(dim, frame): return self.bezier_func(dim, frame, r_func)
    #
    #     def tracker(p0, t_p0, t):
    #         c = np.array([center[0], center[1], p0[2]])
    #         return c + (p0 - c) * r_func(t) / r_func(t_p0)
    #     return self.generate_data(frame_count, cylinder, tracker)
