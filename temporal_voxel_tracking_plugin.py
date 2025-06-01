import qt
import scipy.interpolate

import temporal_voxel_tracking as tvt
import data_generator as dg
import frame_generator as fg
from qt import QTimer
import slicer
from slicer.util import getNode, getNodes
from slicer.ScriptedLoadableModule import *
import numpy as np
import file_interface as fi
from dvc.spline_interpolation import *
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
        id_track_point = 'Algorithm Deduction'
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

        points_menu = main_menu_bar.addMenu("Points")
        points_menu.setObjectName("PointsMenu")

        abaqus_menu = main_menu_bar.addMenu("Abaqus")
        abaqus_menu.setObjectName("AbaqusMenu")

        load_menu = main_menu_bar.addMenu("Load")
        load_menu.setObjectName("LoadMenu")

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

        self.add_action(dvc_menu, "Track Multiple Points", self.on_dvc_track_multiple_points_action_triggered)
        self.add_action(dvc_menu, "Test tracking - points B", self.on_dvc_test_point_action_triggered)
        self.add_action(dvc_menu, "Test tracking - Pulse", self.on_dvc_test_pulse_action_triggered)
        self.add_action(dvc_menu, "Test tracking - Noise Pulse", self.on_dvc_test_noise_pulse_action_triggered)
        self.add_action(dvc_menu, "Perform resolution tests", self.on_dvc_test_resolution)

        self.add_action(points_menu, "Export Selected Markers", self.on_import_selected_markers_action_triggered)

        self.add_action(abaqus_menu, "Import Abaqus simulation", self.on_load_abaqus_simulation)
        self.add_action(abaqus_menu, "Test with Simulation", self.on_test_with_simulation)
        self.add_action(abaqus_menu, "Perform resolution test with simulation", self.on_resolution_test_with_simulation)

        self.add_action(load_menu, "Load dvc test", self.on_load_dvc_test)

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

    def on_dvc_track_multiple_points_action_triggered(self):
        marker = sh.getSelectedMarker()
        if not marker:
            return
        ras2ijk = sh.getRAS2IJKMatrixForFirstAvailableVolume()
        points = sh.getPointsCoords(marker, ras2ijk)
        slicer.mrmlScene.RemoveNode(marker)
        frames, currentFrame = sh.getFramesFromFirstAvailable()
        config = (31, False, 'linear')

        results = np.zeros((len(frames)*len(points), 3))
        for i in range(len(points)):
            point = points[i]
            result, time = self.temporal_voxel_tracking_engine.dvcTrackPoint(point, frames, currentFrame, config)
            for j in range(len(result)):
                results[i*len(result) + j] = result[j]
            print(f"Done: {i} / {len(points)}")

        ijk2ras = sh.getIJK2RASMatrixForFirstAvailableVolume()
        node = sh.createMarkers(results, ijk2ras, 'Algorithm Deduction', '')
        node.GetDisplayNode().SetTextScale(0)


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

    def generate_transform(self, transformType, frameCount, isCyclical):
       startFrame, ijkToRas = sh.getFirstFrameFromData()
       frames = self.frame_generator.generateFrames(startFrame, frameCount, transformType, isCyclical)
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
        self.generate_transform(0, 4, False)

    @staticmethod
    def pulsate(x, t, dim, total):
        pulse = (np.sin(t / (total - 1) * np.pi) ** 2 * 0.5 + 1)
        return (x - dim / 2) * pulse + dim / 2

    def on_generate_pulsation(self):
        self.generate_transform(1, 7, True)

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
        self.generate_transform(2, frames, False)

    def on_dvc_test_pulse_action_triggered(self):
        self.dvc_test_legacy(self.pulsate)

    def on_dvc_test_noise_pulse_action_triggered(self):
        if self.perlin_noise is None:
            print("Perlin noise not initialized!")
            return
        self.dvc_test_legacy(self.pulsate_with_noise)

    def dvc_test(self, frames, frames_coeffs, currentFrame, startingPoints, config):
        results = np.zeros((len(frames) * len(startingPoints), 3))
        times = np.zeros((len(startingPoints)))
        correlations = np.zeros(((len(frames)-1) * len(startingPoints)))
        for i in range(len(startingPoints)):
            point = startingPoints[i]
            result, time, correlation = self.temporal_voxel_tracking_engine.dvcTrackPoint(point, frames, frames_coeffs, currentFrame, config)
            times[i] = time
            for j in range(len(result)):
                results[i * len(result) + j] = result[j]
            for j in range(len(correlation)):
                correlations[i * len(correlation) + j] = correlation[j]
            print(f"Done: {i+1} / {len(startingPoints)}")

        return results, times, correlations

    def dvc_test_legacy2(self, transform, frames, currentFrame, startingPoints, config):
        results = np.zeros((len(frames) * len(startingPoints), 3))
        resultsGT = np.zeros((len(frames) * len(startingPoints), 3))
        times = np.zeros((len(startingPoints)))
        for i in range(len(startingPoints)):
            point = startingPoints[i]
            points, pointsGT, time = self.temporal_voxel_tracking_engine.dvcTrackPointWithGT(point, transform, frames, currentFrame, config)
            times[i] = time
            for j in range(len(points)):
                results[i * len(points) + j] = points[j]
                resultsGT[i * len(pointsGT) + j] = pointsGT[j]
            print(f"Done: {i+1} / {len(startingPoints)}")

        return results, resultsGT, times

    def dvc_test_legacy(self, transform):
        marker = sh.getSelectedMarker()
        if not marker:
            return
        ras2ijk = sh.getRAS2IJKMatrixForFirstAvailableVolume()
        markerPoints = sh.getPointsCoords(marker, ras2ijk)
        slicer.mrmlScene.RemoveNode(marker)
        frames, currentFrame = sh.getFramesFromFirstAvailable()
        config = (31, False, 'linear')

        results, resultsGT, times = self.dvc_test_legacy2(transform, frames, currentFrame, markerPoints, config)

        fi.savePointsWithGT(results, resultsGT, [], len(markerPoints), config, times)
        ijk2ras = sh.getIJK2RASMatrixForFirstAvailableVolume()
        node = sh.createMarkers(results, ijk2ras, 'Algorithm Deduction', '')
        node.GetDisplayNode().SetTextScale(0)

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

    def on_import_selected_markers_action_triggered(self):
        todo()
        # marker = sh.getSelectedMarker()
        # ras2ijk = sh.getRAS2IJKMatrixForFirstAvailableVolume()
        # point = sh.getPointCoords(marker, ras2ijk)
        pass

    def on_dvc_test_resolution(self):
        # normalizedRadius = 0.5
        # resolution = 100
        # halfResolution = resolution // 2
        # radius = halfResolution * normalizedRadius
        # dimensions = (resolution, resolution, resolution)
        # sphere = self.data_generator.generate_sphere(dimensions, radius)
        # animation = self.frame_generator.generateFrames(sphere, 10, 1, True)
        # sh.createSequence(np.array(animation))
        # return

        resolution = 256
        frames = fi.loadArray(f"test\\resolution_test\\{resolution}\\sphere_{resolution}.npy")
        sh.createSequence(frames)
        points, pointsGT = fi.loadPointsWithGT(f'test\\resolution_test\\{resolution}\\result_sphere_{resolution}.csv')
        sh.createMarkers(points, None, 'Algorithm Deduction', '')
        sh.createMarkers(pointsGT, None, 'Ground Truth', '')
        return

        normalizedRadius = 0.5

        numPoints = 5
        np.random.seed(123)
        startingPoints = np.random.randn(numPoints, 3)
        startingPoints /= np.linalg.norm(startingPoints, axis=1)[:, np.newaxis]
        startingPoints *= normalizedRadius

        transformType = 1
        transform = self.pulsate

        resolutions = (64, 256)#203, 232, 256)
        windows = (31, 31)
        # resolutions = (161, 203, 232, 256)
        for resolution, window in zip(resolutions, windows):
            print(f'Testing resolution {resolution}')
            halfResolution = resolution // 2
            radius = halfResolution * normalizedRadius
            dimensions = (resolution, resolution, resolution)

            points = startingPoints * halfResolution + halfResolution

            frames = fi.loadArray(f"test\\resolution_test\\{resolution}\\sphere_{resolution}.npy")
            print(f"Trying to load sphere_{resolution}.npy")
            if frames is None:
                print(f"Generating sphere_{resolution}.npy")
                sphere = self.data_generator.generate_sphere(dimensions, radius)
                animation = self.frame_generator.generateFrames(sphere, 10, transformType, True)
                frames = np.array(animation)
                fi.saveArray(f"test\\resolution_test\\{resolution}\\sphere_{resolution}.npy", frames)
                print(f"Generated and saved sphere_{resolution}.npy")

            print(f"Starting tracking tests...")
            config = (window, False, 'linear')
            results, resultsGT, times = self.dvc_test_legacy2(transform, frames, 0, points, config)
            fi.savePointsWithGT(results, resultsGT, [], len(points), config, times, f'test\\resolution_test\\{resolution}\\result_sphere_{resolution}.csv')
            print(f"Saved result_sphere_{resolution}.csv")

        # sh.createSequence(animation)
        # sh.createMarkers(results, None, "Alg", "")

    def on_load_abaqus_simulation(self):
        dim = 256
        data_path = 'abaqus\\coords\\hiper_elastic.npy'
        simulation_data = fi.loadArray(data_path)
        frames = self.load_abaqus_simulation(simulation_data, dim)
        sh.createSequence(frames)

    def load_abaqus_simulation(self, dim):
        relDir = 'abaqus\\coords\\hiper_elastic.npy'
        data = fi.loadArray(relDir)

        minX = data[0, 1, 0]
        minY = data[0, 2, 0]
        minZ = data[0, 3, 0]
        maxX = data[0, 1, 0]
        maxY = data[0, 2, 0]
        maxZ = data[0, 3, 0]
        for i in range(data.shape[0]):
            vec = data[i, 1:, 0]
            if (minX > vec[0]):
                minX = vec[0]
            if (minY > vec[1]):
                minY = vec[1]
            if (minZ > vec[2]):
                minZ = vec[2]
            if (maxX < vec[0]):
                maxX = vec[0]
            if (maxY < vec[1]):
                maxY = vec[1]
            if (maxZ < vec[2]):
                maxZ = vec[2]
        maxCoord = max(maxX, maxY, maxZ)
        minCoord = min(minX, minY, minZ)

        frames = []
        for frame in range(data.shape[2]):
            print(f"Interpolating frame: {frame}")
            for i in range(data.shape[0]):
                vec = data[i, 1:, frame]
                vec = (vec - minCoord) / (maxCoord - minCoord)
                data[i, 1:, frame] = vec

            grid_x, grid_y, grid_z = np.mgrid[0:1:complex(dim), 0:1:complex(dim), 0:1:complex(dim)]
            points = data[:, 1:4, frame]
            values = data[:, 0, frame]

            maxValue = np.max(values)
            values = (values + 1) / (maxValue + 1)

            frame0 = scipy.interpolate.griddata(points, values, (grid_x, grid_y, grid_z), fill_value=0.0, rescale=False,
                                                method='linear')
            frames.append(frame0)
        return frames

    def create_starters(self, data, dim):
        points = []
        points_edge_count = 2
        points_distance = 0.15
        first_point = (1 - (points_edge_count - 1) * points_distance) / 2
        for i in range(points_edge_count):
            for j in range(points_edge_count):
                for k in range(points_edge_count):
                    x = np.repeat(first_point, 3) + np.array([i, j, k]) * points_distance
                    x *= dim
                    points.append(x)

        # For each marker find the closest data point
        starters = []
        ground_truths = []
        data_count = data.shape[0]
        data_points = data[:, 1:, 0]
        for point in points:
            point = np.array(point) / np.array(dim)

            def key_func(key):
                return np.linalg.norm(point - data_points[key, :])

            index_min = min(range(data_count), key=key_func)
            starters.append(data_points[index_min, :] * np.array(dim))
            ground_truth = np.transpose(data[index_min, 1:, :])
            for i in range(ground_truth.shape[0]):
                ground_truth[i, :] = ground_truth[i, :] * np.array(dim)
            ground_truths.append(ground_truth)
        ground_truths = np.vstack(ground_truths)
        return starters, ground_truths

    def on_test_with_simulation(self):
        # Usual setup for the test
        ijk2ras = sh.getIJK2RASMatrixForFirstAvailableVolume()
        frames, currentFrame = sh.getFramesFromFirstAvailable()
        dim = frames[0].shape
        config = (31, False, 'linear')

        # Load simulation
        relDir = 'abaqus\\coords\\hiper_elastic.npy'
        data = fi.loadArray(relDir)

        # Get existing points
        # marker = sh.getSelectedMarker()
        # if not marker:
        #     return
        # ras2ijk = sh.getRAS2IJKMatrixForFirstAvailableVolume()
        # points = sh.getPointsCoords(marker, ras2ijk)
        # slicer.mrmlScene.RemoveNode(marker)
        # OR
        # Create points on grid
        starters, ground_truths = self.create_starters(data, dim)

        results, times, correlations = self.dvc_test(frames, currentFrame, starters, config)

        # sh.createMarkers(starters, ijk2ras, "Closest Starters", "S")
        sh.createMarkers(results, ijk2ras, "Algorithm Deduction", "AD-")
        sh.createMarkers(ground_truths, ijk2ras, "Ground Truth", "GT-")

        fi.savePointsWithGT(results, ground_truths, correlations, len(starters), config, times, f'abaqus\\multiple_test.csv')
        print(f'abaqus\\multiple_test.csv')

    # def on_resolution_test_with_simulation(self):
    #     frames, current_frame = sh.getFramesFromFirstAvailable()
    #     n = 4
    #     frames = frames[0, ::n, ::n, ::n]
    #
    #
    #     # frames_coeffs = np.zeros((frames.shape[0], frames.shape[1] + 6, frames.shape[2] + 6, frames.shape[3] + 6))
    #     # for frame in range(frames.shape[0]):
    #     #     frames_coeffs[frame] = calculate_spline5_coefficients(frames[frame, :, :, :])
    #
    #     coeffs = calculate_spline5_coefficients(frames)
    #     factor = 4
    #     sx, sy, s10 = np.meshgrid(np.linspace(0, frames.shape[0] - 1, frames.shape[0] * factor),
    #                               np.linspace(0, frames.shape[1] - 1, frames.shape[1] * factor),
    #                               np.linspace(0, frames.shape[2] - 1, frames.shape[2] * factor), indexing='ij')
    #     sz = interpolate_spline5(np.array([sx, sy, s10]), coeffs)
    #
    #     frames = np.repeat(np.repeat(np.repeat(frames,factor, axis=0), factor, axis=1), factor, axis=2)
    #     sh.createSequence([frames])
    #     sh.createSequence([sz])
    #     result = self.temporal_voxel_tracking_engine.vt.dvc.find_correlated_point(frames, sz, np.array([48,48,48]), 31, False, "linear")
    #     print(result)
    #     result = self.temporal_voxel_tracking_engine.vt.dvc.find_correlated_point(frames, sz, np.array([96,48,48]), 31, False, "linear")
    #     print(result)
    #     result = self.temporal_voxel_tracking_engine.vt.dvc.find_correlated_point(frames, sz, np.array([48,96,48]), 31, False, "linear")
    #     print(result)
    #     result = self.temporal_voxel_tracking_engine.vt.dvc.find_correlated_point(frames, sz, np.array([48,48,96]), 31, False, "linear")
    #     print(result)

    def on_resolution_test_with_simulation(self):
        data_path = 'abaqus\\coords\\hiper_elastic.npy'
        simulation_data = fi.loadArray(data_path)

        # resolutions = (32, 64)  # 203, 232, 256)
        windows = (31, 31, 31)
        resolutions = (161, 203, 232)
        interpolation = "spline5"
        for resolution, window in zip(resolutions, windows):
            print(f'Testing resolution {resolution}')
            cube_path = f"abaqus\\hiper_elastic_scene\\{resolution}\\cube_{resolution}.npy"
            frames = fi.loadArray(cube_path)
            # frames = np.array([frames[0], frames[frames.shape[0]-1]])  # TEST limit to 2 only
            print(f"Trying to load cube_{resolution}.npy")
            if frames is None:
                print(f"Generating cube_{resolution}.npy")
                frames = np.array(self.load_abaqus_simulation(resolution))
                fi.saveArray(cube_path, frames)
                print(f"Generated and saved cube_{resolution}.npy")
            frames_coeffs = np.zeros((frames.shape[0], frames.shape[1]+6, frames.shape[2]+6, frames.shape[3]+6))
            for frame in range(frames.shape[0]):
                frames_coeffs[frame] = calculate_spline5_coefficients(frames[frame, :, :, :])
            #
            # sh.createSequence(frames_coeffs[:, 4:-3, 4:-3, 4:-3])
            #
            # slice = frames[0, :, :, :]
            # # print(slice.shape)
            # # ix, iy = np.indices(slice.shape)
            # factor = 4
            # sx, sy, s10 = np.meshgrid(np.linspace(0, slice.shape[0] - 1, slice.shape[0] * factor),
            #                           np.linspace(0, slice.shape[1] - 1, slice.shape[1] * factor),
            #                           np.linspace(0, slice.shape[2] - 1, slice.shape[2] * factor), indexing='ij')
            # sz = interpolate_spline5(np.array([sx, sy, s10]), frames_coeffs[0])
            # # sx = sx[:,:,0]
            # # sy = sy[:,:,0]
            # # sz = sz[:,:,0]
            # # #xyz_vectors = np.stack([x, sy, szs], axis=-1)
            # #
            # slice = np.repeat(np.repeat(np.repeat(slice,factor, axis=0), factor, axis=1), factor, axis=2)
            # sh.createSequence([slice])
            # sh.createSequence([sz])
            # result = self.temporal_voxel_tracking_engine.vt.dvc.find_correlated_point(slice, sz, np.array([48,48,48]), 31, False, "linear")
            # print(result)
            # slice = [slice[i] for i in range(slice.shape[0])]

            # Now use the density values from the slice as z coordinates
            # iz = slice[ix, iy]
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            # ax.scatter(ix.flatten(), iy.flatten(), iz.flatten(), c=iz.flatten(), cmap='viridis', marker='o')
            # # ax.plot_surface(sx, sy, sz, linewidth=0, cmap=cm.coolwarm, antialiased=False)
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Density')
            # root_path = f'C:\\Users\\USER\\Documents\\Repositories\\temporal-voxel-tracking\\'
            # savefig(root_path + "11.png")

            # interpolate_spline5(frames[0, :, :, :], np.array([10, 10, 10]), coeffs)

            print(f"Starting tracking tests...")
            config = (window, False, interpolation)
            starters, ground_truths = self.create_starters(simulation_data, resolution)
            results, times, correlations = self.dvc_test(frames, frames_coeffs,0, starters, config)
            fi.savePointsWithGT(results, ground_truths, correlations, len(starters), config, times,
                                f'abaqus\\hiper_elastic_scene\\{resolution}\\result_cube_{resolution}.csv')
            print(f"Saved result_cube_{resolution}.csv")

    def on_load_dvc_test(self):
        resolution = 128
        frames = fi.loadArray(f"abaqus\\hiper_elastic_scene\\{resolution}\\cube_{resolution}.npy")
        sh.createSequence(frames)
        points, pointsGT = fi.loadPointsWithGT(f'abaqus\\hiper_elastic_scene\\{resolution}\\result_cube_{resolution}.csv')
        ijk2ras = sh.getIJK2RASMatrixForFirstAvailableVolume()
        sh.createMarkers(points, ijk2ras, "Algorithm Deduction", "AD-")
        sh.createMarkers(pointsGT, ijk2ras, "Ground Truth", "GT-")


# ------------------- MAIN -------------------

# Instantiate the custom menu action
plugin = TemporalVoxelTrackingPlugin(slicer.util.mainWindow())
