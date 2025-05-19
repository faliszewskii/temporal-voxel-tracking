from odbAccess import openOdb

root_path = f'C:\\Users\\USER\\Documents\\Repositories\\temporal-voxel-tracking\\'
relPath = 'abaqus_odb\\comp\\Job-1.odb'

path = root_path + relPath
odb = openOdb(path=path)
step = odb.steps.values()[0]  # change to your step name
frame = step.frames[-1]     # last frame (final result)

instance = odb.rootAssembly.instances['PART-1-1']  # change to your instance name

for node in instance.nodes:
    print(f"Node {node.label}: {node.coordinates}")
