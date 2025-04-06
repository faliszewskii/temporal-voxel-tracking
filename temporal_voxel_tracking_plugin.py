import qt
import temporal_voxel_tracking as tvt
import data_generator as dg
import frame_generator as fg
from qt import QTimer
import slicer
from slicer.util import getNode, getNodes
from slicer.ScriptedLoadableModule import *
import numpy as np
import file_interface as fi
from perlin_noise import PerlinNoise
from scipy.interpolate import RegularGridInterpolator

import slicer_helpers as sh


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
        self.perlin_noise: RegularGridInterpolator = None

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

        generate_menu = main_menu_bar.addMenu("Generate Data")
        generate_menu.setObjectName("GenerateMenu")

        generate_frames_menu = main_menu_bar.addMenu("Generate Frames")
        generate_frames_menu.setObjectName("GenerateFramesMenu")

        noise_menu = main_menu_bar.addMenu("Noise")
        noise_menu.setObjectName("NoiseMenu")

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

        self.add_action(generate_frames_menu, "Generate Rotating Frames", self.on_generate_rotation)
        self.add_action(generate_frames_menu, "Generate Pulsating Frames", self.on_generate_pulsation)
        self.add_action(generate_frames_menu, "Generate Noise Pulsating Frames", self.on_generate_pulsation_with_noise)

        self.add_action(noise_menu, "Generate Perlin Noise For Current Volume", self.on_generate_perlin_noise)
        self.add_action(noise_menu, "Load Perlin Noise", self.on_load_perlin_noise)

        self.add_action(dvc_menu, "Track Point", self.on_dvc_track_point_action_triggered)
        self.add_action(dvc_menu, "Test tracking - points B", self.on_dvc_test_point_action_triggered)
        self.add_action(dvc_menu, "Test tracking - Pulse", self.on_dvc_test_pulse_action_triggered)
        self.add_action(dvc_menu, "Test tracking - Noise Pulse", self.on_dvc_test_noise_pulse_action_triggered)

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

    def generate_transform(self, transform, frameCount, isCyclical):
       startFrame, ijkToRas = sh.getFirstFrameFromData()
       frames = self.frame_generator.generateFrames(startFrame, frameCount, transform, isCyclical)
       sh.createSequence(frames, ijkToRas)

    @staticmethod
    def rotate(x, t, dim, total):
        rot = np.array([
            [1, 0, 0],
            [0, np.cos(0.3*t), -np.sin(0.3*t)],
            [0, np.sin(0.3*t), np.cos(0.3*t)]
        ])
        return rot @ (x-dim/2) + dim/2

    def on_generate_rotation(self):
        self.generate_transform(self.rotate, 4, False)

    @staticmethod
    def pulsate(x, t, dim, total):
        pulse = (np.sin(t / (total - 1) * np.pi) ** 2 * 0.5 + 1)
        return (x - dim / 2) * pulse + dim / 2

    def on_generate_pulsation(self):
        self.generate_transform(self.pulsate, 7, True)

    def pulsate_with_noise(self, x, t, dim, total):
        noiseMovement = np.array([10, 0, 0]) * t
        noiseStrength = 4
        pulse = (np.sin(t / (total - 1) * np.pi) ** 2 * 0.3 + 1)
        newX = (x - dim / 2) * pulse + dim / 2
        noise = self.perlin_noise((newX + noiseMovement) % dim)[0] * noiseStrength
        return newX + noise

    def on_generate_pulsation_with_noise(self):
        if self.perlin_noise is None:
            print("Perlin noise not initialized!")
            return
        frames = 9
        self.generate_transform(self.pulsate_with_noise, frames, False)

    def on_dvc_test_pulse_action_triggered(self):
        self.dvc_test(self.pulsate)

    def on_dvc_test_noise_pulse_action_triggered(self):
        if self.perlin_noise is None:
            print("Perlin noise not initialized!")
            return
        self.dvc_test(self.pulsate_with_noise)

    def dvc_test(self, transform):
        marker = sh.getSelectedMarker()
        if not marker:
            return
        ras2ijk = sh.getRAS2IJKMatrixForFirstAvailableVolume()
        point = sh.getPointCoords(marker, ras2ijk)
        slicer.mrmlScene.RemoveNode(marker)
        frames, currentFrame = sh.getFramesFromFirstAvailable()
        config = (31, False, 'linear')
        points, pointsGT, time = self.temporal_voxel_tracking_engine.dvcTrackPointWithGT(point, transform, frames, currentFrame, config)
        fi.savePointWithGT(points, pointsGT, config, time)
        ijk2ras = sh.getIJK2RASMatrixForFirstAvailableVolume()
        sh.createMarkers(points, ijk2ras, 'Algorithm Deduction', '')
        sh.createMarkers(pointsGT, ijk2ras, 'Ground Truth', 'GT')

    def initialize_perlin_noise(self, dim):
        print("Generating noise")

        noise1 = PerlinNoise(octaves=10, seed=1)
        noise2 = PerlinNoise(octaves=10, seed=2)
        noise3 = PerlinNoise(octaves=10, seed=3)
        tensor_field = np.zeros((dim[0], dim[1], dim[2], 3))  # 3 for the 3 noise functions

        for x in range(dim[0]):
            for y in range(dim[1]):
                for z in range(dim[2]):
                    point = [x / dim[0], y / dim[1], z / dim[2]]  # Normalized coordinates in [0, 1]
                    noiseVector = np.array([noise1(point), noise2(point), noise3(point)])
                    tensor_field[x, y, z] = noiseVector

        return tensor_field

    def on_generate_perlin_noise(self):
        dim = sh.getFirstFrameDim()
        perlin_noise = self.initialize_perlin_noise(dim)
        fi.saveArray(f'test\\perlin_noise\\vector_perlin_noise_{dim[0]}_{dim[1]}_{dim[2]}.npy', perlin_noise)

        range_x = np.array(range(0, dim[0]))
        range_y = np.array(range(0, dim[1]))
        range_z = np.array(range(0, dim[2]))
        self.perlin_noise = RegularGridInterpolator((range_x, range_y, range_z), perlin_noise, fill_value=0.0, bounds_error=False)

        print("Perlin noise initialized")

    def on_load_perlin_noise(self):
        dim = sh.getFirstFrameDim()
        perlin_noise = fi.loadArray(f'test\\perlin_noise\\vector_perlin_noise_{dim[0]}_{dim[1]}_{dim[2]}.npy')

        range_x = np.array(range(0, dim[0]))
        range_y = np.array(range(0, dim[1]))
        range_z = np.array(range(0, dim[2]))
        self.perlin_noise = RegularGridInterpolator((range_x, range_y, range_z), perlin_noise, fill_value=0.0, bounds_error=False)

        print("Perlin noise loaded")


# ------------------- MAIN -------------------

# Instantiate the custom menu action
plugin = TemporalVoxelTrackingPlugin(slicer.util.mainWindow())
