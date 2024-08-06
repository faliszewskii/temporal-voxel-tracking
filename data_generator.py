import slicer
import vtk
import numpy as np
from vtk.util import numpy_support
import random


class DataGenerator:

    def __init__(self):
        pass

    def generate_data(self, frame_count, data_func):
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
            data_vtk = numpy_support.numpy_to_vtk(num_array=data_func(dimensions, i).ravel(), deep=True, array_type=vtk.VTK_FLOAT)

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
        self.generate_data(frame_count, lambda dimensions, i: self.linear_cube_data(dimensions, i, [f(x) for x in np.arange(frame_count)]))

    def generate_slow_cube(self):
        print("Generating slow cube...")
        frame_count = 10
        def f(x): return x * np.array([1.0, 0.0, 0.0])
        self.generate_data(frame_count, lambda dimensions, i: self.linear_cube_data(dimensions, i, [f(x) for x in np.arange(frame_count)]))

    def generate_faster_cube(self):
        print("Generating faster cube...")
        frame_count = 10
        def f(x): return x * np.array([2.0, 0.0, 0.0])
        self.generate_data(frame_count, lambda dimensions, i: self.linear_cube_data(dimensions, i, [f(x) for x in np.arange(frame_count)]))

    def generate_random_cube(self):
        print("Generating random cube...")
        frame_count = 10
        current_pos = [0.0, 0.0, 0.0]
        def f(x): return 10 * np.array([random.random(), random.random(), random.random()]) + current_pos
        self.generate_data(frame_count, lambda dimensions, i: self.linear_cube_data(dimensions, i, [f(x) for x in np.arange(frame_count)]))
