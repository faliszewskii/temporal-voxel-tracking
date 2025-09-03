import os

import numpy as np
from odbAccess import openOdb
from symbolicConstants import *
from abaqusConstants import *

root_path = f'C:\\Users\\USER\\Documents\\Repositories\\temporal-voxel-tracking\\'
# root_path = f'C:\\temp\\'


def extract_deformed_positions(odbPath, inpPath):
    """
    Extracts the deformed nodal positions from the Abaqus ODB file.

    Args:
    odb_path (str): Path to the ODB file.
    frame_indices (list): Indices of frames to extract. If None, extracts all frames.

    Returns:
    np.ndarray: Numpy array of shape (num_nodes, 3, num_frames) where:
                 - num_nodes: Total number of nodes in the model.
                 - 3: X, Y, Z positions.
                 - num_frames: Total number of frames extracted.
    """
    path = root_path + odbPath
    odb = openOdb(path)

    # Beware. It only handles one part models !!!

    instanceNames = list()

    for instance in odb.rootAssembly.instances.values():
        instanceNames += [instance.name]
    print(instanceNames)

    path = root_path + inpPath
    mdb.ModelFromInputFile(name='MyModel', inputFileName=path)

    # Now you can access parts, materials, etc.
    model = mdb.models['MyModel']
    part = model.parts.values()[0]
    material_names = model.materials.keys()
    print("Materials:", material_names)
    print(part.sets)

    assembly = model.rootAssembly
    instance = assembly.instances.values()[0]

    node_material_map = {}
    for i in range(len(part.sets.items())):
        # if 'Set-' in elset_name:
        #     continue  # skip auto-generated sets
        # try:
        setName, set = part.sets.items()[i]
        print(f'Set to find: {setName}')
        material = 0
        for j in range(len(part.sectionAssignments)):
            assignment = part.sectionAssignments[j]
            print(assignment.region)
            assignedSet = assignment.region[0]  # Get only name
            if setName == assignedSet:
                material = j
        for node in set.nodes:
            node_material_map[node.label] = material


    all_frames = odb.steps.values()[0].frames
    frame_indices = range(len(all_frames))

    deformed_positions = []
    for frame_index in frame_indices:
        frame = all_frames[frame_index]
        displacement = frame.fieldOutputs['COORD']
        nodes = displacement.values

        frame_positions = []

        for node in nodes:
            if node.position != NODAL:
                continue
            # if node.instance.name != "COMPOSITE-1":
            #     continue
            # print(f'{node.position}, {node.elementLabel}, {node.nodeLabel}, {node.instance.name}')

            node_coords = node.data[0], node.data[1], node.data[2]
            frame_positions.append(node_coords[:])

        deformed_positions.append(np.array(frame_positions))

    deformed_positions = np.stack(deformed_positions, axis=2)

    odb.close()
    # print(instanceNames)

    return deformed_positions


# Example Usage
odb_path = 'abaqus_odb\\regular80\\regular80.odb'
# odb_path = 'Job-1.odb'
inp_path = 'abaqus_odb\\regular80\\regular80.inp'
# inp_path = 'Job-1.inp'
deformed_positions = extract_deformed_positions(odb_path, inp_path)

print(deformed_positions.shape)
relDir = 'abaqus\\coords\\regular80.npy'
array = deformed_positions

path = root_path + relDir
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, 'wb') as f:
    np.save(f, array)
print(f"Saved to: {relDir}")