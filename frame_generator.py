import numpy as np


class FrameGenerator:
    def __init__(self):
        pass

    def generateTranslation(self, startFrame, frameCount):
        def translate(x, t):
            return x + np.array([t, 0, 0])

        return self.generateFrames(startFrame, frameCount, translate)

    def generateFrames(self, startFrame, frameCount, transform):
        dim = startFrame.shape
        tetrahedra = [
            [[0, 1, 1], [1, 1, 1], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1]],
            [[1, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 0, 0], [0, 1, 0]],
            [[1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]]
        ]

        transformedFrame = startFrame
        for nx in range(dim[0]):
            for ny in range(dim[1]):
                for nz in range(dim[2]):
                    transformedFrame[nx, ny, nz] = transform(nx, ny, nz)

        for nx in range(dim[0]):
            for ny in range(dim[1]):
                for nz in range(dim[2]):
                    # For each tetrahedron get its bounding box and try to assign values to the integer numbered cells
                    # that fit inside the bounding box
                    for tetra in tetrahedra:
                        p0 = transformedFrame[nx + tetra[0][0], ny + tetra[0][1], nz + tetra[0][2]]
                        p1 = transformedFrame[nx + tetra[1][0], ny + tetra[1][1], nz + tetra[1][2]]
                        p2 = transformedFrame[nx + tetra[2][0], ny + tetra[2][1], nz + tetra[2][2]]
                        p3 = transformedFrame[nx + tetra[3][0], ny + tetra[3][1], nz + tetra[3][2]]
                        (minX, maxX, minY, maxY, minZ, maxZ) = self.getBoundingBox(p0, p1, p2, p3)

        pass

    def getBoundingBox(self, p0, p1, p2, p3):

        pass
