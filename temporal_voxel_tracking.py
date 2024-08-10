import numpy as np
import time
import slicer
import math
import vtk
from vtk.util import numpy_support
from slicer.util import getNode, getNodes
import optical_flow as of

# Identifiers
id_start_tracking_point = 'start_tracking_point'
id_track_point = 'track_point'
id_start_tracking_volume = 'start_tracking_volume'


def create_volume(i, cube_size, volume_node):
    cubeVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
    cubeVolume.SetName(f'CubeVolume_{i}')

    ijk_to_ras_matrix = getIJK2RASMatrix()
    # Define the dimensions of the volume
    dimensions = volume_node.GetImageData().GetDimensions()
    spacing = volume_node.GetSpacing()
    origin = volume_node.GetOrigin()

    # Set the origin, spacing, and dimensions of the volume
    cubeVolume.SetOrigin(origin)
    cubeVolume.SetSpacing(spacing)
    # cubeVolume.SetDimensions(dimensions)

    # Create an array to hold the volume data
    cubeArray = np.zeros(dimensions, dtype=np.int16)

    # Define the size of the cube
    half_cube_size = cube_size // 2

    # Calculate the center of the volume
    center = [dim // 2 for dim in dimensions]

    start_x = max(center[0] - half_cube_size, 0)
    end_x = min(center[0] + half_cube_size, dimensions[0])
    start_y = max(center[1] - half_cube_size, 0)
    end_y = min(center[1] + half_cube_size, dimensions[1])
    start_z = max(center[2] - half_cube_size, 0)
    end_z = min(center[2] + half_cube_size, dimensions[2])

    # Fill the array with the cube values
    cubeArray[start_x:end_x, start_y:end_y, start_z:end_z] = 1

    # Set the volume data
    slicer.util.updateVolumeFromArray(cubeVolume, cubeArray)

    # Add the volume to the scene
    # slicer.mrmlScene.AddNode(cubeVolume)

    cubeVolume.CreateDefaultDisplayNodes()
    cubeVolume.CreateDefaultStorageNode()


def start_tracking_volume():
    # fiducial_node = getNode(id_start_tracking_volume)
    # starting_coords = getPointCoords(fiducial_node)
    # slicer.mrmlScene.RemoveNode(fiducial_node)

    volume_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')
    [current_frame, frame_count] = getFrames()
    for i in range(frame_count):
        create_volume(i, (i + 1) * 10, volume_node)


def invert_image(input_volume):
    vector_image = vtk.vtkImageData()
    vector_image.SetDimensions(input_volume.GetDimensions()[0], input_volume.GetDimensions()[1],input_volume.GetDimensions()[2])
    vector_image.AllocateScalars(vtk.VTK_DOUBLE, 9)

    # Iterate through the image and invert the pixel values
    for z in range(vector_image.GetDimensions()[2]):
        for y in range(vector_image.GetDimensions()[1]):
            for x in range(vector_image.GetDimensions()[0]):
                tensor = [1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0]

                # Set the tensor for each voxel
                for i in range(9):
                    vector_image.SetScalarComponentFromDouble(x, y, z, i, tensor[i])

    return vector_image


def create_transform_from_optical_flow(optical_flow_data):
    shape = optical_flow_data.shape
    nx, ny, nz, _ = shape
    n_points = nx * ny * nz

    # Create the displacement field array
    displacement_array = vtk.vtkDoubleArray()
    displacement_array.SetNumberOfComponents(3)
    displacement_array.SetNumberOfTuples(n_points)

    # Fill the displacement array with vectors from optical flow data
    idx = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                vector = optical_flow_data[i, j, k]
                displacement_array.SetTuple(idx, vector)
                idx += 1

    # Create a vtkImageData to store the displacements
    grid = vtk.vtkImageData()
    grid.SetDimensions(nx, ny, nz)
    grid.SetSpacing(1.0, 1.0, 1.0)
    grid.SetOrigin(0, 0, 0)
    grid.GetPointData().SetVectors(displacement_array)

    # Create a grid transform
    grid_transform = vtk.vtkGridTransform()
    grid_transform.SetDisplacementGridData(grid)
    grid_transform.Update()

    # Create a transform node and set the grid transform
    transform_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode")
    transform_node.SetName("OpticalFlowTransform")
    transform_node.SetAndObserveTransformFromParent(grid_transform)

    return transform_node


def display_transform(transform_node):
    # Create a transform display node
    display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformDisplayNode")
    display_node.SetName("OpticalFlowTransformDisplay")
    display_node.SetVisibility(True)

    # Set display properties
    # display_node.SetGlyphScale(1.0)  # Adjust the scale as needed
    # display_node.SetGlyphSpacing(1.0)  # Set spacing between glyphs
    # display_node.SetGlyphTypeToArrow()
    # display_node.SetGlyphSize(0.5)  # Size of the arrows
    # display_node.SetVisibility2DVectorField(True)  # Show in 2D views
    # display_node.SetVisibility3DVectorField(True)  # Show in 3D views

    # Set color by magnitude (optional)
    # display_node.SetColorModeToMagnitude()
    # display_node.SetActiveColorArrayName("Magnitude")

    # Link the display node to the transform node
    transform_node.AddAndObserveDisplayNodeID(display_node.GetID())


class TemporalVoxelTrackingEngine:
    def __init__(self):
        self.optical_flow_sequence = []

    def create_displacement_map(self):
        self.optical_flow_sequence.clear()
        sequence_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSequenceNode')

        frames = sequence_node.GetNumberOfDataNodes()
        # frames_minus_1 = 3
        for index in range(frames):
            print(f'{index}:', end='')
            # Get the current and next volume
            current_volume = sequence_node.GetNthDataNode(index)
            next_i = index + 1
            if next_i == sequence_node.GetNumberOfDataNodes():
                next_i = 0
            next_volume = sequence_node.GetNthDataNode(next_i)

            dims = current_volume.GetImageData().GetDimensions()

            current_array = numpy_support.vtk_to_numpy(current_volume.GetImageData().GetPointData().GetScalars())
            next_array = numpy_support.vtk_to_numpy(next_volume.GetImageData().GetPointData().GetScalars())

            current_array = current_array.reshape((dims[0], dims[1], dims[2]), order='F')
            next_array = next_array.reshape((dims[0], dims[1], dims[2]), order='F')

            optical_flow_data = of.compute_optical_flow(current_array, next_array)
            self.optical_flow_sequence.append(optical_flow_data)

    def display_magnitude_volume(self, origin_volume):
        original_sequence_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSequenceNode')

        new_sequence_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode")
        new_sequence_node.SetName("Optical Flow Magnitude Sequence")
        new_sequence_node.SetIndexName(original_sequence_node.GetIndexName())
        new_sequence_node.SetIndexUnit(original_sequence_node.GetIndexUnit())

        seq_browser = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode")
        seq_browser.SetName("Optical Flow Magnitude Browser")
        seq_browser.SetAndObserveMasterSequenceNodeID(new_sequence_node.GetID())
        seq_browser.SetSaveChanges(new_sequence_node, True)

        for index, volume_data in enumerate(self.optical_flow_sequence):

            vector_image = vtk.vtkImageData()
            vector_image.SetDimensions(volume_data.shape[0], volume_data.shape[1], volume_data.shape[2])  # Set the dimensions (adjust as needed)
            vector_image.AllocateScalars(vtk.VTK_DOUBLE, 1)  # 3 components for a 3D vector

            # Iterate over each voxel and set vector values
            for z in range(vector_image.GetDimensions()[2]):
                for y in range(vector_image.GetDimensions()[1]):
                    for x in range(vector_image.GetDimensions()[0]):
                        # Define a simple vector (replace with actual vector data as needed)
                        v = volume_data[x, y, z]
                        norm = np.linalg.norm(v)  # Example vector data
                        # Set the vector for each voxel
                        vector_image.SetScalarComponentFromDouble(x, y, z, 0, norm)

            # Create a scalar volume node to hold the vector data
            vector_volume_node = slicer.vtkMRMLScalarVolumeNode()
            vector_volume_node.SetName(f"Optical Flow {index}")
            vector_volume_node.SetAndObserveImageData(vector_image)

            # Set origin, spacing, and IJK to RAS direction matrix (adjust as needed)
            ijkToRasMatrix = vtk.vtkMatrix4x4()
            origin_volume.GetIJKToRASMatrix(ijkToRasMatrix)
            vector_volume_node.SetIJKToRASDirectionMatrix(ijkToRasMatrix)
            vector_volume_node.SetOrigin(origin_volume.GetOrigin())
            vector_volume_node.SetSpacing(origin_volume.GetSpacing())

            # Set up the display node for the vector volume (optional)
            # vector_display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeDisplayNode")
            # slicer.mrmlScene.AddNode(vector_display_node)
            # vector_volume_node.SetAndObserveDisplayNodeID(vector_display_node.GetID())

            new_sequence_node.SetDataNodeAtValue(vector_volume_node, original_sequence_node.GetNthIndexValue(index))
            # sequence_node.SetDataNodeAtValue(vector_volume_node, sequence_node.GetNthIndexValue(index))

    def display_vector_field(self, volume_node, step=2, scale=1.0):
        """
        Create a sequence of vector field nodes for each time point, aligned with a DICOM volume.

        Args:
            vector_fields: A list of 3D numpy arrays, where each array contains 3D vectors for a time point.
            volume_node: The volume node containing the DICOM data.
            scale: A scaling factor for the arrows.
            step: The step size for displaying vectors, e.g., every 10th vector.
        """

        original_sequence_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSequenceNode')

        sequence_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode")
        sequence_node.SetName("Optical Flow Vectors Sequence")
        sequence_node.SetIndexName(original_sequence_node.GetIndexName())
        sequence_node.SetIndexUnit(original_sequence_node.GetIndexUnit())

        seq_browser = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode")
        seq_browser.SetName("Optical Flow Vectors Browser")
        seq_browser.SetAndObserveMasterSequenceNodeID(sequence_node.GetID())
        seq_browser.SetSaveChanges(sequence_node, True)

        # Get IJK to RAS transformation matrix
        ijkToRasMatrix = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(ijkToRasMatrix)

        for t, vector_field in enumerate(self.optical_flow_sequence):
            N, M, P, _ = vector_field.shape

            # Create a polydata for the current time point
            points = vtk.vtkPoints()
            vectors = vtk.vtkFloatArray()
            vectors.SetNumberOfComponents(3)
            polydata = vtk.vtkPolyData()
            magnitudes = []

            # Fill in the points and vectors
            for i in range(0, N, step):
                for j in range(0, M, step):
                    for k in range(0, P, step):
                        vector = vector_field[i, j, k]
                        magnitude = np.linalg.norm(vector)
                        if magnitude <= 0.001:
                            continue
                        # IJK coordinates (voxel indices)
                        ijk = [i, j, k, 1.0]

                        # Transform IJK to RAS
                        ras = [0, 0, 0, 1.0]
                        ijkToRasMatrix.MultiplyPoint(ijk, ras)

                        # Add the point and vector to the polydata
                        points.InsertNextPoint(ras[0:3])
                        vectors.InsertNextTuple(vector)

                        # Calculate magnitude for color and scale
                        magnitudes.append(magnitude)

            # Set the points and vectors in the polydata
            polydata.SetPoints(points)
            polydata.GetPointData().SetVectors(vectors)

            # Calculate range of magnitudes
            min_mag = min(magnitudes)
            max_mag = max(magnitudes)

            # Create color mapping based on magnitudes
            lookup_table = vtk.vtkLookupTable()
            lookup_table.SetRange(min_mag, max_mag)
            lookup_table.Build()

            # Create a magnitude array for coloring
            magnitude_array = vtk.vtkFloatArray()
            magnitude_array.SetName("Magnitude")
            for magnitude in magnitudes:
                magnitude_array.InsertNextValue(magnitude)

            # Create an arrow source
            arrowSource = vtk.vtkArrowSource()
            arrowSource.SetTipResolution(10)
            arrowSource.SetTipRadius(0.1)
            arrowSource.SetTipLength(2)
            arrowSource.SetShaftResolution(10)
            arrowSource.SetShaftRadius(0.05)

            # Create a glyph 3D
            glyph3D = vtk.vtkGlyph3D()
            glyph3D.SetSourceConnection(arrowSource.GetOutputPort())
            glyph3D.SetInputData(polydata)
            glyph3D.SetVectorModeToUseVector()
            glyph3D.SetColorModeToColorByScalar()
            glyph3D.ScalingOn()
            glyph3D.SetScaleFactor(scale)
            glyph3D.OrientOn()
            glyph3D.Update()

            # Add this polydata as a new data node in the sequence
            # Add glyph data to a model node
            model_node = slicer.vtkMRMLModelNode()
            model_node.SetAndObservePolyData(glyph3D.GetOutput())
            model_node.SetName(f"VectorField_{t}")

            sequence_node.SetDataNodeAtValue(model_node, original_sequence_node.GetNthIndexValue(t))

            # Create and set display node
            # display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
            # display_node.SetColor(1, 0, 0)  # Set color (red here)
            # display_node.SetVisibility(True)
            # model_node.SetAndObserveDisplayNodeID(display_node.GetID())

            slicer.app.processEvents()


        slicer.app.processEvents()

    def getIJK2RASMatrix(self):
        volume_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')
        if not volume_node:
            return None

        ijk_to_ras_matrix = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(ijk_to_ras_matrix)
        return ijk_to_ras_matrix

    def getRAS2IJKMatrix(self):
        volume_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')
        if not volume_node:
            return None

        ras_to_ijk_matrix = vtk.vtkMatrix4x4()
        volume_node.GetRASToIJKMatrix(ras_to_ijk_matrix)
        return ras_to_ijk_matrix

    def changeBasis(self, coords, basis_matrix):
        source = [coords[0], coords[1], coords[2], 1]
        destination = [0, 0, 0, 0]
        basis_matrix.MultiplyPoint(source, destination)
        return destination[:3]

    def generateTrack(self, starting_coords, current_frame, frame_count, optical_flow_frames):
        ijk_to_ras_matrix = self.getIJK2RASMatrix()
        markups_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        markups_node.SetName(id_track_point)
        markups_node.GetDisplayNode().SetSelectedColor(1, 0, 0)
        markups_node.GetDisplayNode().SetColor(1, 1, 0)

        points = np.zeros((frame_count, 3))

        # Current frame
        points[current_frame] = np.array(starting_coords)

        # Previous frames
        track_vector = np.array(starting_coords)
        best_vector = track_vector.copy()
        # Previous frames
        for i in range(current_frame):
            t = current_frame - i - 1
            optical_flow = optical_flow_frames[t]
            minDist = float("inf")
            for x in range(optical_flow.shape[0]):
                for y in range(optical_flow.shape[1]):
                    for z in range(optical_flow.shape[2]):
                        optical_flow_value = optical_flow[x][y][z]
                        next_vector = track_vector - np.array(optical_flow_value)
                        dist = np.linalg.norm(next_vector - np.array([x, y, z]))
                        if dist < minDist:
                            minDist = dist
                            best_vector = next_vector
            track_vector = best_vector
            points[t] = track_vector

        # Forward frames
        forward_vector = np.array(starting_coords)
        for i in range(frame_count - current_frame - 1):
            t = (i + current_frame)
            optical_flow = optical_flow_frames[t][int(math.floor(forward_vector[0]))][int(math.floor(forward_vector[1]))][int(math.floor(forward_vector[2]))]
            forward_vector += np.array(optical_flow)
            points[t + 1] = forward_vector

        for i in range(len(points)):
            coords = self.changeBasis(points[i], ijk_to_ras_matrix)
            markups_node.AddControlPoint(*coords)
            markups_node.SetNthControlPointLabel(i, f"{i}")

        display_node = markups_node.GetDisplayNode()
        display_node.SetOccludedVisibility(True)
        display_node.SetOccludedOpacity(0.6)

        return points

    def getFrames(self):
        sequence_browser_nodes = slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode')
        if not sequence_browser_nodes:
            print("No sequence browser nodes found")
            return

        sequence_browser_node = sequence_browser_nodes[0]  # Assume the first one, refine as needed
        sequence_node = sequence_browser_node.GetMasterSequenceNode()
        if not sequence_node:
            print("No master sequence node found in the sequence browser")
            return None

        current_frame = sequence_browser_node.GetSelectedItemNumber()
        frame_count = sequence_browser_node.GetNumberOfItems()

        return current_frame, frame_count

    def getPointCoords(self, fiducial_node):
        ras_to_ijk_matrix = self.getRAS2IJKMatrix()
        if not ras_to_ijk_matrix:
            print("No volume detected to track")
            return None

        coords = [0.0, 0.0, 0.0]
        fiducial_node.GetNthControlPointPosition(0, coords)
        return self.changeBasis(coords, ras_to_ijk_matrix)

    def compare_to_truth(self, points, tracker, starting_coords, current_frame):
        sum_error = 0
        min_error = float("inf")
        max_error = float("-inf")
        for i in range(len(points)):
            error = (np.linalg.norm(points[i] - tracker(starting_coords, current_frame, i))) ** 2
            sum_error += error
            if error > max_error:
                max_error = error
            if error < min_error:
                min_error = error
        print(f"Error: {sum_error}, min: {min_error}, max: {max_error}")

        ijk_to_ras_matrix = self.getIJK2RASMatrix()
        markups_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        markups_node.SetName("Ground Truth")
        markups_node.GetDisplayNode().SetSelectedColor(0, 1, 1)
        markups_node.GetDisplayNode().SetColor(0, 1, 0)
        for i in range(len(points)):
            coords = self.changeBasis(tracker(starting_coords, current_frame, i), ijk_to_ras_matrix)
            markups_node.AddControlPoint(*coords)
            markups_node.SetNthControlPointLabel(i, f"GT:{i}")

        display_node = markups_node.GetDisplayNode()
        display_node.SetOccludedVisibility(True)
        display_node.SetOccludedOpacity(0.6)

    def start_tracking_point(self, fiducial_node, tracker):
        starting_coords = self.getPointCoords(fiducial_node)
        if not starting_coords:
            return
        slicer.mrmlScene.RemoveNode(fiducial_node)
        if getNodes(id_track_point, None):
            slicer.mrmlScene.RemoveNode(getNode(id_track_point))
        [current_frame, frame_count] = self.getFrames()
        points = self.generateTrack(starting_coords, current_frame, frame_count, self.optical_flow_sequence)
        self.compare_to_truth(points, tracker, starting_coords, current_frame)

    def track_point(self):
        [current_frame, frame_count] = self.getFrames()
        fiducial_node = getNode(id_track_point)
        for i in range(frame_count):
            fiducial_node.SetNthControlPointSelected(i, i == current_frame)
