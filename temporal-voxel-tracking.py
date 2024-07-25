from qt import QTimer
import numpy as np
import time
import slicer
import math

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


def generateTrack(starting_coords, current_frame, frame_count):
    ijk_to_ras_matrix = getIJK2RASMatrix()
    markups_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
    markups_node.SetName(id_track_point)
    markups_node.GetDisplayNode().SetSelectedColor(1, 0, 0)
    markups_node.GetDisplayNode().SetColor(1, 1, 0)
    step = 1
    h = 15
    alpha = 0.8
    r = 20
    for i in range(frame_count):
        t = (i - current_frame) * step
        x = starting_coords[0] + r - r * math.cos(alpha * t)
        y = starting_coords[1] + r * math.sin(alpha * t)
        z = starting_coords[2] + h * t
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


def start_tracking_point(fiducial_node):
    starting_coords = getPointCoords(fiducial_node)
    if not starting_coords:
        return
    slicer.mrmlScene.RemoveNode(fiducial_node)
    if getNodes(id_track_point, None):
        slicer.mrmlScene.RemoveNode(getNode(id_track_point))
    [current_frame, frame_count] = getFrames()
    generateTrack(starting_coords, current_frame, frame_count)


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
        self.track_mesh_action = qt.QAction("Create Displacement Map", self.parent)
        self.track_mesh_action.triggered.connect(self.on_track_mesh_action_triggered)

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

    @staticmethod
    def on_track_point_action_triggered():
        if not slicer.util.getNode('vtkMRMLSelectionNodeSingleton').GetActivePlaceNodeID():
            print("no node selected")
            return
        selected_node = slicer.mrmlScene.GetNodeByID(
            slicer.util.getNode('vtkMRMLSelectionNodeSingleton').GetActivePlaceNodeID())
        if selected_node.IsA('vtkMRMLMarkupsFiducialNode'):
            start_tracking_point(selected_node)

    @staticmethod
    def on_track_volume_action_triggered():
        start_tracking_volume()

    @staticmethod
    def on_track_mesh_action_triggered():
        todo()


def updateFrame():
    if getNodes(id_track_point, None) and getNode(id_track_point).IsA('vtkMRMLMarkupsFiducialNode'):
        track_point()

    [current_frame, frame_count] = getFrames()
    for i in range(frame_count):
        if not getNodes(f'CubeVolume_{i}', None):
            continue
        node = getNode(f'CubeVolume_{i}')
        vrDisplayNode = slicer.modules.volumerendering.logic().CreateDefaultVolumeRenderingNodes(node)
        vrDisplayNode.SetVisibility(i == current_frame)
        # Display the volume
        # slicer.util.setSliceViewerLayers(background=getNode(f'CubeVolume_{current_frame}'))
        # vrDisplayNode.SetShowMode(slicer.vtkMRMLDisplayNode.ShowIgnore)


# ------------------- MAIN -------------------

# Instantiate the custom menu action
custom_menu_action = CustomMenuAction(slicer.util.mainWindow())

# Qt timer
timer = QTimer()
timer.setInterval(20)
timer.timeout.connect(updateFrame)
timer.start()
