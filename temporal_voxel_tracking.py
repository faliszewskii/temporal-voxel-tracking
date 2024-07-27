from qt import QTimer
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


def getIJK2RASMatrix():
    volume_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')
    if not volume_node:
        return None

    ijk_to_ras_matrix = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(ijk_to_ras_matrix)
    return ijk_to_ras_matrix


def getRAS2IJKMatrix():
    volume_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')
    if not volume_node:
        return None

    ras_to_ijk_matrix = vtk.vtkMatrix4x4()
    volume_node.GetRASToIJKMatrix(ras_to_ijk_matrix)
    return ras_to_ijk_matrix


def changeBasis(coords, basis_matrix):
    source = [coords[0], coords[1], coords[2], 1]
    destination = [0, 0, 0, 0]
    basis_matrix.MultiplyPoint(source, destination)
    return destination[:3]


def generateTrack(starting_coords, current_frame, frame_count, optical_flow_frames):
    ijk_to_ras_matrix = getIJK2RASMatrix()
    markups_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
    markups_node.SetName(id_track_point)
    markups_node.GetDisplayNode().SetSelectedColor(1, 0, 0)
    markups_node.GetDisplayNode().SetColor(1, 1, 0)
    h = 15
    alpha = 0.8
    r = 20
    x = starting_coords[0]
    y = starting_coords[1]
    z = starting_coords[2]
    coords = changeBasis([x, y, z], ijk_to_ras_matrix)
    markups_node.AddControlPoint(*coords)
    current_frame = 0  # TODO # Assume first
    for i in range(frame_count - 1):
        t = (i - current_frame)
        optical_flow = optical_flow_frames[t][int(math.floor(x))][int(math.floor(y))][int(math.floor(z))]
        x += optical_flow[0]
        y += optical_flow[1]
        z += optical_flow[2]
        # x = starting_coords[0] + r - r * math.cos(alpha * t)
        # y = starting_coords[1] + r * math.sin(alpha * t)
        # z = starting_coords[2] + h * t
        coords = changeBasis([x, y, z], ijk_to_ras_matrix)
        markups_node.AddControlPoint(*coords)

    for i in range(markups_node.GetNumberOfControlPoints()):
        markups_node.SetNthControlPointLabel(i, f"{i}")

    display_node = markups_node.GetDisplayNode()
    display_node.SetOccludedVisibility(True)
    display_node.SetOccludedOpacity(0.6)


def getFrames():
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


def getPointCoords(fiducial_node):
    ras_to_ijk_matrix = getRAS2IJKMatrix()
    if not ras_to_ijk_matrix:
        print("No volume detected to track")
        return None

    coords = [0.0, 0.0, 0.0]
    fiducial_node.GetNthControlPointPosition(0, coords)
    return changeBasis(coords, ras_to_ijk_matrix)


def start_tracking_point(fiducial_node, optical_flow_frames, frames = -1):
    starting_coords = getPointCoords(fiducial_node)
    if not starting_coords:
        return
    slicer.mrmlScene.RemoveNode(fiducial_node)
    if getNodes(id_track_point, None):
        slicer.mrmlScene.RemoveNode(getNode(id_track_point))
    [current_frame, frame_count] = getFrames()
    if frames == -1:
        frames = frame_count
    generateTrack(starting_coords, current_frame, frames, optical_flow_frames)


def track_point():
    [current_frame, frame_count] = getFrames()
    fiducial_node = getNode(id_track_point)
    for i in range(frame_count):
        fiducial_node.SetNthControlPointSelected(i, i == current_frame)


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


def display_vector_field(vector_fields, volume_node, scale=2.0, step=20):
    """
    Create a sequence of vector field nodes for each time point, aligned with a DICOM volume.

    Args:
        vector_fields: A list of 3D numpy arrays, where each array contains 3D vectors for a time point.
        volume_node: The volume node containing the DICOM data.
        scale: A scaling factor for the arrows.
        step: The step size for displaying vectors, e.g., every 10th vector.
    """

    # Create the sequence node to hold vector fields at each time point
    sequence_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode")
    sequence_node.SetName("VectorFieldSequence")

    # Get IJK to RAS transformation matrix
    ijkToRasMatrix = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(ijkToRasMatrix)

    for t, vector_field in enumerate(vector_fields):
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
                    # IJK coordinates (voxel indices)
                    ijk = [i, j, k, 1.0]

                    # Transform IJK to RAS
                    ras = [0, 0, 0, 1.0]
                    ijkToRasMatrix.MultiplyPoint(ijk, ras)

                    # Add the point and vector to the polydata
                    vector = vector_field[i, j, k]
                    points.InsertNextPoint(ras[0:3])
                    vectors.InsertNextTuple(vector)

                    # Calculate magnitude for color and scale
                    magnitude = np.linalg.norm(vector)
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
        arrowSource.SetTipRadius(0.5)
        arrowSource.SetTipLength(2)
        arrowSource.SetShaftResolution(10)
        arrowSource.SetShaftRadius(0.2)

        # Create a glyph 3D
        glyph3D = vtk.vtkGlyph3D()
        glyph3D.SetSourceConnection(arrowSource.GetOutputPort())
        glyph3D.SetInputData(polydata)
        glyph3D.SetVectorModeToUseVector()
        glyph3D.SetColorModeToColorByScalar()
        glyph3D.ScalingOn()
        glyph3D.SetScaleFactor(scale)
        glyph3D.OrientOn()
        # glyph3D.SetLookupTable(lookup_table)
        glyph3D.Update()

        # Add this polydata as a new data node in the sequence
        # Add glyph data to a model node
        model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        model_node.SetAndObservePolyData(glyph3D.GetOutput())
        model_node.SetName(f"VectorField_{t}")

        # Create and set display node
        display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        display_node.SetColor(1, 0, 0)  # Set color (red here)
        display_node.SetVisibility(True)
        model_node.SetAndObserveDisplayNodeID(display_node.GetID())

        # Optionally, adjust the scale factor for visual clarity
        # display_node.SetVisibility2DFill(False)
        # display_node.SetVisibility2DOutline(False)

        slicer.app.processEvents()

        # Add model node to the sequence
        # sequence_node.SetDataNodeAtValue(model_node, str(t))

    # # Create the sequence browser node
    # sequence_browser_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode")
    # sequence_browser_node.SetName("VectorFieldSequenceBrowser")
    # sequence_browser_node.AddSynchronizedSequenceNodeID(sequence_node.GetID())
    # sequence_browser_node.SetPlaybackActive(True)
    # sequence_browser_node.SetRecordingActive(False)
    # sequence_browser_node.SetPlaybackRateFps(1.0)  # Set the playback rate (frames per second)
    #
    # # Set the display properties
    # display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
    # display_node.SetColor(0, 1, 0)  # Set color (green here)
    # display_node.SetVisibility(True)
    # # display_node.SetScale(scale)
    #
    # # Connect the display node to each model node
    # for t in range(sequence_node.GetNumberOfDataNodes()):
    #     model_node = sequence_node.GetNthDataNode(t)
    #     model_node.SetAndObserveDisplayNodeID(display_node.GetID())

    slicer.app.processEvents()


def display_magnitude_volume(volume_datas, origin_volume):
    original_sequence_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSequenceNode')

    new_sequence_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode")
    new_sequence_node.SetName("Optical Flow Sequence")
    new_sequence_node.SetIndexName(original_sequence_node.GetIndexName())
    new_sequence_node.SetIndexUnit(original_sequence_node.GetIndexUnit())

    seq_browser = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode")
    seq_browser.SetAndObserveMasterSequenceNodeID(new_sequence_node.GetID())
    seq_browser.SetSaveChanges(new_sequence_node, True)

    for index, volume_data in enumerate(volume_datas):

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
        vector_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"Optical Flow {index}")
        vector_volume_node.SetAndObserveImageData(vector_image)

        # Set origin, spacing, and IJK to RAS direction matrix (adjust as needed)
        ijkToRasMatrix = vtk.vtkMatrix4x4()
        origin_volume.GetIJKToRASMatrix(ijkToRasMatrix)
        vector_volume_node.SetIJKToRASDirectionMatrix(ijkToRasMatrix)
        vector_volume_node.SetOrigin(origin_volume.GetOrigin())
        vector_volume_node.SetSpacing(origin_volume.GetSpacing())

        # Set up the display node for the vector volume (optional)
        vector_display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeDisplayNode")
        slicer.mrmlScene.AddNode(vector_display_node)
        vector_volume_node.SetAndObserveDisplayNodeID(vector_display_node.GetID())

        new_sequence_node.SetDataNodeAtValue(vector_volume_node, original_sequence_node.GetNthIndexValue(index))
        # sequence_node.SetDataNodeAtValue(vector_volume_node, sequence_node.GetNthIndexValue(index))

def create_displacement_map(input_volume):
    sequence_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSequenceNode')
    optical_flow = []

    # frames_minus_1 = sequence_node.GetNumberOfDataNodes() - 1
    frames_minus_1 = 3
    for index in range(frames_minus_1):
        # Get the current and next volume
        current_volume = sequence_node.GetNthDataNode(index)
        next_volume = sequence_node.GetNthDataNode(index + 1)

        dims = current_volume.GetImageData().GetDimensions()

        current_array = numpy_support.vtk_to_numpy(current_volume.GetImageData().GetPointData().GetScalars())
        next_array = numpy_support.vtk_to_numpy(next_volume.GetImageData().GetPointData().GetScalars())

        current_array = current_array.reshape(dims[2], dims[1], dims[0])
        next_array = next_array.reshape(dims[2], dims[1], dims[0])

        optical_flow_data = of.compute_optical_flow(current_array, next_array)
        optical_flow.append(optical_flow_data)
    # Display the result
    # display_magnitude_volume(optical_flow, input_volume)
    return optical_flow

def get_selected_node():
    active_place_node_id = slicer.util.getNode('vtkMRMLSelectionNodeSingleton').GetActivePlaceNodeID()
    if not active_place_node_id:
        return None
    return slicer.mrmlScene.GetNodeByID(active_place_node_id)


class CustomMenuAction:
    def __init__(self, parent=None):
        self.parent = parent  # Parent widget (usually the main window)
        self.create_actions()
        self.add_actions_to_menu_bar()

    def create_actions(self):
        # Create a new QAction
        self.track_point_action = qt.QAction("Track Point", self.parent)
        self.track_point_action.triggered.connect(self.on_track_point_action_triggered)
        self.track_volume_action = qt.QAction("Track Volume", self.parent)
        self.track_volume_action.triggered.connect(self.on_track_volume_action_triggered)
        self.track_mesh_action = qt.QAction("Track Mesh", self.parent)
        self.track_mesh_action.triggered.connect(self.on_track_mesh_action_triggered)
        self.displacement_map_action = qt.QAction("Create Displacement Map", self.parent)
        self.displacement_map_action.triggered.connect(self.on_create_displacement_map_action_triggered)

    def add_actions_to_menu_bar(self):
        # Get the main menu bar from the parent widget
        main_menu_bar = self.parent.menuBar()

        # Find or create a menu to add the action to
        custom_menu = main_menu_bar.findChild(qt.QMenu, "CustomMenu")
        if not custom_menu:
            custom_menu = main_menu_bar.addMenu("Track")
            custom_menu.setObjectName("CustomMenu")  # Set an object name for identification

        # Add the action to the menu
        custom_menu.addAction(self.track_point_action)
        custom_menu.addAction(self.track_volume_action)
        custom_menu.addAction(self.track_mesh_action)
        custom_menu.addAction(self.displacement_map_action)

    @staticmethod
    def on_track_point_action_triggered():
        optial_flow = []
        selected_node = slicer.mrmlScene.GetNodeByID(slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetBackgroundLayer().GetVolumeNode().GetID())
        if selected_node and selected_node.IsA('vtkMRMLScalarVolumeNode'):
            optical_flow = create_displacement_map(selected_node)
        else:
            print("no scalar volume selected")
            return
        selected_node = get_selected_node()
        if selected_node and selected_node.IsA('vtkMRMLMarkupsFiducialNode'):
            start_tracking_point(selected_node, optical_flow)
        else:
            print("no node selected")

    @staticmethod
    def on_track_volume_action_triggered():
        start_tracking_volume()

    @staticmethod
    def on_track_mesh_action_triggered():
        todo()

    @staticmethod
    def on_create_displacement_map_action_triggered():
        selected_node = slicer.mrmlScene.GetNodeByID(slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetBackgroundLayer().GetVolumeNode().GetID())
        if selected_node and selected_node.IsA('vtkMRMLScalarVolumeNode'):
            optical_flow = create_displacement_map(selected_node)
            display_magnitude_volume(optical_flow, selected_node)
        else:
           print("no scalar volume selected")



def updateFrame():
    if getNodes(id_track_point, None) and getNode(id_track_point).IsA('vtkMRMLMarkupsFiducialNode'):
        track_point()

    # [current_frame, frame_count] = getFrames()
    # for i in range(frame_count):
    #     if not getNodes(f'CubeVolume_{i}', None):
    #         continue
    #     node = getNode(f'CubeVolume_{i}')
    #     vrDisplayNode = slicer.modules.volumerendering.logic().CreateDefaultVolumeRenderingNodes(node)
    #     vrDisplayNode.SetVisibility(i == current_frame)
    #     # Display the volume
    #     # slicer.util.setSliceViewerLayers(background=getNode(f'CubeVolume_{current_frame}'))
    #     # vrDisplayNode.SetShowMode(slicer.vtkMRMLDisplayNode.ShowIgnore)


# ------------------- MAIN -------------------

# Instantiate the custom menu action
custom_menu_action = CustomMenuAction(slicer.util.mainWindow())

# Qt timer
timer = QTimer()
timer.setInterval(20)
timer.timeout.connect(updateFrame)
timer.start()
