from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from job import *
from sketch import *
import numpy as np


# model
my_model = mdb.models["Model-1"]

# box
my_sketch = my_model.ConstrainedSketch(name='box', sheetSize=2.0)
my_sketch.rectangle(point1=(0.0, 0.0), point2=(1.0, 1.0))
part_box = my_model.Part(dimensionality=THREE_D, name="box", type=DEFORMABLE_BODY)
part_box.BaseSolidExtrude(depth=1.0, sketch=my_sketch)

#assemebly
assembly = my_model.rootAssembly
assembly.DatumCsysByDefault(CARTESIAN)
inc_box = assembly.Instance(dependent=ON, name='box', part=part_box)


# data
number_of_inclusions = 30
min_gap = 0.02
it = 1
it_max = 100
radius  = lambda:   0.05 + np.random.rand()*0.15
data = []
x1 = np.random.rand()
y1 = np.random.rand()
z1 = np.random.rand()
r1 = radius()
data.append([x1,y1,z1, r1])
run = True
while run:
	it +=1
	r1= radius()
	x1 = np.random.rand()
	y1 = np.random.rand()
	z1 = np.random.rand()
	for d in data:
		x2, y2, z2, r2 = d
		distance = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2) - r1 -r2
		if distance<min_gap:break
	else:
		data.append([x1,y1,z1, r1])
	if len(data)==number_of_inclusions: run = False
	if it>it_max:
		run=False
		print("error")

# inclusion loop
inclusion_list = []
for i, d in enumerate(data):
	x, y, z, radius = d
	part_name = "inclusion_{:d}".format(i+1)
	inc_name = "inc_{:d}".format(i+1)
	s1 = my_model.ConstrainedSketch(name='inclusion', sheetSize=2.0)
	s1.ConstructionLine(point1=(0.0, -1.0), point2=(0.0, 1.0))
	s1.FixedConstraint(entity=s1.geometry[2])
	s1.ArcByCenterEnds(center=(0.0, 0.0), direction=CLOCKWISE,
	 point1=(0.0, radius), point2=(0.0, -radius))
	s1.Line(point1=(0.0, radius), point2=(0.0, -radius))
	part_inclusion=my_model.Part(dimensionality=THREE_D,
	 name=part_name, type=DEFORMABLE_BODY)
	part_inclusion.BaseSolidRevolve(angle=360.0,
	 flipRevolveDirection=OFF, sketch=s1)
	inc = assembly.Instance(dependent=ON, name=inc_name,
	 part=part_inclusion)
	assembly.translate(instanceList=(inc_name, ), vector=(x, y, z))
	inclusion_list.append(inc)

# merge inclusions
inclusions = assembly.InstanceFromBooleanMerge(domain=GEOMETRY,
	instances=inclusion_list, name='inclusions',
	originalInstances=DELETE)

#cut matrix
matrix = assembly.InstanceFromBooleanCut(
 cuttingInstances=(inclusions,),
 instanceToBeCut=inc_box,
 name='matrix',
 originalInstances=DELETE)

# cutting instance
my_sketch = my_model.ConstrainedSketch(name='c1', sheetSize=2.0)
my_sketch.rectangle(point1=(0,0), point2=(3, 3))
c1 = my_model.Part(dimensionality=THREE_D, name="c1",
	type=DEFORMABLE_BODY)
c1.BaseSolidExtrude(depth=3.0, sketch=my_sketch)
c1 = assembly.Instance(dependent=ON, name='c1', part=c1)
assembly.translate(instanceList=("c1", ), vector=(-1, -1, -1))

# box with box
inc_box = assembly.Instance(dependent=ON, name='box', part=part_box)
c2 = assembly.InstanceFromBooleanCut(cuttingInstances=(inc_box,),
	instanceToBeCut=c1, name='c1',originalInstances=DELETE)


#cut inclusions
inclusions = assembly.Instance(dependent=ON, name='inclusion',
	part=my_model.parts["inclusions"])
inclusions = assembly.InstanceFromBooleanCut(cuttingInstances=(c2,),
	instanceToBeCut=inclusions, name='inclusions',originalInstances=DELETE)


composite = assembly.InstanceFromBooleanMerge(domain=GEOMETRY,
	instances=(matrix, inclusions), name='composite',
	originalInstances=DELETE, keepIntersections=ON)



# execfile(r"C:\Users\Admin\Dropbox\www\data\abaqus\cae\abaqus_10_composite.py")