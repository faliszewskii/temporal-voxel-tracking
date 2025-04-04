import slicer
from slicer.ScriptedLoadableModule import *
import numpy as np
from vtk.util import numpy_support
import vtk


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