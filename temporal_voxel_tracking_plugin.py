import qt
import temporal_voxel_tracking as tvt
import data_generator as dg
import frame_generator as fg
from qt import QTimer
import slicer
from slicer.util import getNode, getNodes
from slicer.ScriptedLoadableModule import *
import numpy as np
from vtk.util import numpy_support
import vtk

class TemporalVoxelTrackingPlugin:
    def __init__(self, parent=None):
        self.current_data = None
        self.temporal_voxel_tracking_engine = tvt.TemporalVoxelTrackingEngine()
        self.data_generator = dg.DataGenerator()
        self.parent = parent  # Parent widget (usually the main window)
        self.actions = []
        self.add_actions_to_menu_bar()
        self.run_timer()
        self.frame_generator = fg.FrameGenerator()

    def update_frame(self):
        id_track_point = 'track_point'
        if getNodes(id_track_point, None) and getNode(id_track_point).IsA('vtkMRMLMarkupsFiducialNode'):
            self.temporal_voxel_tracking_engine.track_point()

    def run_timer(self):
        self.timer = QTimer()
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()

    def add_action(self, menu, name, callback):
        self.actions.append(qt.QAction(name, self.parent))
        self.actions[-1].triggered.connect(callback)
        menu.addAction(self.actions[-1])

    def add_actions_to_menu_bar(self):
        main_menu_bar = self.parent.menuBar()

        optical_flow_menu = main_menu_bar.addMenu("Optical Flow")
        optical_flow_menu.setObjectName("TrackMenu")
        generate_menu = main_menu_bar.addMenu("Generate")
        generate_menu.setObjectName("GenerateMenu")
        dvc_menu = main_menu_bar.addMenu("DVC")
        dvc_menu.setObjectName("DVCMenu")

        self.add_action(optical_flow_menu, "Calculate optical flow for selected", self.on_calculate_optical_flow_action_triggered)
        optical_flow_menu.addSeparator()
        self.add_action(optical_flow_menu, "Show Optical Flow Magnitudes", self.on_show_magnitudes_action_triggered)
        self.add_action(optical_flow_menu, "Show Optical Flow Vectors", self.on_show_vectors_action_triggered)
        optical_flow_menu.addSeparator()
        self.add_action(optical_flow_menu, "Track Point", self.on_track_point_action_triggered)
        self.add_action(optical_flow_menu, "Track Volume", self.on_track_volume_action_triggered)
        self.add_action(optical_flow_menu, "Track Mesh", self.on_track_mesh_action_triggered)

        self.add_action(generate_menu, "Generate Static Cube", lambda _: self.generate(self.data_generator.generate_static_cube))
        self.add_action(generate_menu, "Generate Slow Cube", lambda _: self.generate(self.data_generator.generate_slow_cube))
        self.add_action(generate_menu, "Generate Faster Cube", lambda _: self.generate(self.data_generator.generate_faster_cube))
        self.add_action(generate_menu, "Generate Random Cube", lambda _: self.generate(self.data_generator.generate_random_cube))
        self.add_action(generate_menu, "Generate Pulsating Cylinder", lambda _: self.generate(self.data_generator.generate_pulsating_cylinder))
        generate_menu.addSeparator()
        self.add_action(generate_menu, "Generate Translated Frames", lambda _: self.on_generate_translation())

        self.add_action(dvc_menu, "Track Point", self.on_dvc_track_point_action_triggered)
        self.add_action(dvc_menu, "Test tracking - point", self.on_dvc_test_point_action_triggered)

    def generate(self, func):
        self.current_data = func()

    def on_show_magnitudes_action_triggered(self):
        selected_node = slicer.mrmlScene.GetNodeByID(slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetBackgroundLayer().GetVolumeNode().GetID())
        if selected_node and selected_node.IsA('vtkMRMLScalarVolumeNode'):
            self.temporal_voxel_tracking_engine.display_magnitude_volume(selected_node)
        else:
            print("no scalar volume selected")

    def on_show_vectors_action_triggered(self):
        selected_node = slicer.mrmlScene.GetNodeByID(slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetBackgroundLayer().GetVolumeNode().GetID())
        if selected_node and selected_node.IsA('vtkMRMLScalarVolumeNode'):
            self.temporal_voxel_tracking_engine.display_vector_field(selected_node, 5, 5)
        else:
            print("no scalar volume selected")

    def on_calculate_optical_flow_action_triggered(self):
        print("Calculating Optical Flow...")
        self.temporal_voxel_tracking_engine.create_displacement_map()
        print("Done.")
        print(f"Optical flow range: {np.min(self.temporal_voxel_tracking_engine.optical_flow_sequence)}, {np.max(self.temporal_voxel_tracking_engine.optical_flow_sequence)}")

    def on_track_point_action_triggered(self):
        active_place_node_id = slicer.util.getNode('vtkMRMLSelectionNodeSingleton').GetActivePlaceNodeID()
        if not active_place_node_id:
            return None
        selected_node = slicer.mrmlScene.GetNodeByID(active_place_node_id)
        if selected_node and selected_node.IsA('vtkMRMLMarkupsFiducialNode'):
            self.temporal_voxel_tracking_engine.start_tracking_point(selected_node, self.current_data.tracking_function)
        else:
            print("no node selected")

    def on_dvc_track_point_action_triggered(self):
        active_place_node_id = slicer.util.getNode('vtkMRMLSelectionNodeSingleton').GetActivePlaceNodeID()
        if not active_place_node_id:
            return None
        selected_node = slicer.mrmlScene.GetNodeByID(active_place_node_id)
        if selected_node and selected_node.IsA('vtkMRMLMarkupsFiducialNode'):
            self.temporal_voxel_tracking_engine.dvc_track_fiducial_node(selected_node)
        else:
            print("no node selected")

    def on_dvc_test_point_action_triggered(self):
        pointsA = [
            (-28.097206, -10.121669, 37.682655),
            (-30.234774, -8.469118, 33.840611),
            (-31.945389, -4.917310, 28.385954),
            (-32.057339, -3.572445, 26.841396),
            (-31.740509, -3.664992, 27.248146),
            (-31.628189, -5.967639, 29.815594),
            (-31.648130, -8.259072, 32.242516),
            (-30.955448, -9.622770, 34.371704),
            (-28.858099, -10.093299, 36.912762)
        ]
        pointsB = [
            (-30.142559, -10.356653, 31.015547),
            (-29.794868, -10.037838, 30.927279),
            (-31.184607, -3.484786, 24.132965),
            (-31.884451, -7.143429, 27.065422),
            (-30.934578, -2.063097, 22.980680),
            (-31.506128, -2.228712, 22.798498),
            (-32.443405, -3.587524, 23.492443),
            (-31.370991, -9.353150, 29.382631),
            (-31.766310, -6.360522, 26.419477),
            (-32.516121, -5.252571, 24.970381)
        ]
        # pointsC = [
        #     (-41.369629, 89.200050, -0.142498),
        #     (-59.968548, 51.035389, -72.672943)
        # ]
        # self.temporal_voxel_tracking_engine.test_points(pointsA)
        self.temporal_voxel_tracking_engine.test_points(pointsB)
        # self.temporal_voxel_tracking_engine.test_points(pointsC)

    def on_track_volume_action_triggered(self):
        todo()

    def on_track_mesh_action_triggered(self):
        todo()

    def on_generate_translation(self):
        def translate(x, t):
            return x + np.array([t, 0, 0])

        startFrame = self.getFirstFrameFromData()
        frames = self.frame_generator.generateFrames(startFrame, 10, translate)
        self.createSequence(frames)

    def getFirstFrameFromData(self):
        sequence_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSequenceNode')
        current_volume = sequence_node.GetNthDataNode(0)
        dims = current_volume.GetImageData().GetDimensions()
        current_array = numpy_support.vtk_to_numpy(current_volume.GetImageData().GetPointData().GetScalars())
        current_array = current_array.reshape((dims[0], dims[1], dims[2]), order='F')
        return current_array

    def createSequence(self, frames):
        sequence_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode")
        sequence_node.SetName("Generated Data Sequence")
        sequence_node.SetIndexName("frames")
        sequence_node.SetIndexUnit("frame")

        seq_browser = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode")
        seq_browser.SetName("Generated Data Browser")
        seq_browser.SetAndObserveMasterSequenceNodeID(sequence_node.GetID())
        seq_browser.SetSaveChanges(sequence_node, True)

        for i in range(len(frames)):
            data_vtk = numpy_support.numpy_to_vtk(num_array=frames[i].ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

            image_data = vtk.vtkImageData()
            image_data.SetDimensions(frames[i].shape)
            image_data.AllocateScalars(vtk.VTK_FLOAT, 1)
            image_data.GetPointData().SetScalars(data_vtk)

            volume_node = slicer.vtkMRMLScalarVolumeNode()
            volume_node.SetOrigin(0, 0, 0)
            volume_node.SetSpacing(1, 1, 1)
            volume_node.SetIJKToRASDirectionMatrix(vtk.vtkMatrix4x4())
            volume_node.SetAndObserveImageData(image_data)

            sequence_node.SetDataNodeAtValue(volume_node, str(i))

# ------------------- MAIN -------------------

# Instantiate the custom menu action
plugin = TemporalVoxelTrackingPlugin(slicer.util.mainWindow())
