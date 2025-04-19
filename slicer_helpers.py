import slicer
from slicer.ScriptedLoadableModule import *
import numpy as np
from vtk.util import numpy_support
import vtk


def getFirstFrameDim():
    sequence_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSequenceNode')
    current_volume = sequence_node.GetNthDataNode(0)
    dims = current_volume.GetImageData().GetDimensions()
    return dims


def getFirstFrameFromData():
    sequence_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSequenceNode')
    current_volume = sequence_node.GetNthDataNode(0)
    dims = current_volume.GetImageData().GetDimensions()
    current_array = numpy_support.vtk_to_numpy(current_volume.GetImageData().GetPointData().GetScalars())
    current_array = current_array.reshape((dims[0], dims[1], dims[2]), order='F')

    ijk_to_ras_matrix = vtk.vtkMatrix4x4()
    current_volume.GetIJKToRASMatrix(ijk_to_ras_matrix)

    return current_array, ijk_to_ras_matrix


def createSequence(frames, ijkToRas):
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
        volume_node.SetIJKToRASMatrix(ijkToRas)
        volume_node.SetAndObserveImageData(image_data)

        sequence_node.SetDataNodeAtValue(volume_node, str(i))


def getSelectedMarker():
    active_place_node_id = slicer.util.getNode('vtkMRMLSelectionNodeSingleton').GetActivePlaceNodeID()
    if not active_place_node_id:
        return None
    selected_node = slicer.mrmlScene.GetNodeByID(active_place_node_id)
    if selected_node and selected_node.IsA('vtkMRMLMarkupsFiducialNode'):
        return selected_node
    else:
        print("no node selected")
    return None


def getRAS2IJKMatrixForFirstAvailableVolume():
    volume_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')
    if not volume_node:
        print("No volume detected to track")
        return None

    ras2ijk = vtk.vtkMatrix4x4()
    volume_node.GetRASToIJKMatrix(ras2ijk)

    if not ras2ijk:
        print("No ras2ijk available in volume")
        return None
    return ras2ijk

def getIJK2RASMatrixForFirstAvailableVolume():
    volume_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')
    if not volume_node:
        print("No volume detected to track")
        return None

    ijk2ras = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(ijk2ras)

    if not ijk2ras:
        print("No ijk2ras available in volume")
        return None
    return ijk2ras


def changeBasis(coords, basis_matrix):
    source = [coords[0], coords[1], coords[2], 1]
    destination = [0, 0, 0, 0]
    basis_matrix.MultiplyPoint(source, destination)
    return destination[:3]


def getPointsCoords(fiducial_node, ras2ijk):
    points = []
    num_points = fiducial_node.GetNumberOfControlPoints()
    for i in range(num_points):
        coords = [0, 0, 0]
        fiducial_node.GetNthControlPointPosition(i, coords)
        point = changeBasis(coords, ras2ijk)
        points.append(point)
    return points


def getSequenceDataFromFirstAvailable():
    sequence_browser_nodes = slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode')
    if not sequence_browser_nodes:
        print("No sequence browser nodes found")
        return

    sequence_browser_node = sequence_browser_nodes[0]
    sequence_node = sequence_browser_node.GetMasterSequenceNode()
    if not sequence_node:
        print("No master sequence node found in the sequence browser")
        return None

    current_frame = sequence_browser_node.GetSelectedItemNumber()
    frame_count = sequence_browser_node.GetNumberOfItems()

    return current_frame, frame_count


def getFramesFromFirstAvailable():
    sequence_node = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSequenceNode')

    current_frame, frame_count = getSequenceDataFromFirstAvailable()

    arrays = []
    for t in range(frame_count):
        current_volume = sequence_node.GetNthDataNode(t)
        dims = current_volume.GetImageData().GetDimensions()
        current_array = numpy_support.vtk_to_numpy(current_volume.GetImageData().GetPointData().GetScalars())
        current_array = current_array.reshape((dims[0], dims[1], dims[2]), order='F')
        arrays.append(current_array)

    frames = np.stack(arrays, axis=0)
    return frames, current_frame


def createMarkers(points, ijk2ras, name, prefix):
    markups_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
    markups_node.SetName(name)
    markups_node.GetDisplayNode().SetGlyphScale(1.5)
    markups_node.GetDisplayNode().SetSelectedColor(1, 0, 0)
    markups_node.GetDisplayNode().SetColor(1, 1, 0)
    for i in range(len(points)):

        coords = changeBasis(points[i], ijk2ras)
        markups_node.AddControlPoint(*coords)
        markups_node.SetNthControlPointLabel(i, f"{prefix}{i}")

    display_node = markups_node.GetDisplayNode()
    display_node.SetOccludedVisibility(True)
    display_node.SetOccludedOpacity(0.6)

    return markups_node

def getNodesWithName(name):
    matching_nodes = []
    for node in slicer.mrmlScene.GetNodes():
            if node.GetName() == name:
                matching_nodes.append(node)
    return matching_nodes