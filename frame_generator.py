import math

import numpy as np
from scipy.interpolate import barycentric_interpolate


class FrameGenerator:
    def __init__(self):
        self.tetrahedra = [
            [[0, 1, 1], [1, 1, 1], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1]],
            [[1, 1, 1], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 0, 0], [0, 1, 0]],
            [[1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]]
        ]

    def generateFrames(self, startFrame, totalFrameCount, transform):
        frames = [startFrame]

        for t in range(1, totalFrameCount):
            frame = self.generateFrame(startFrame, transform, t)
            frames.append(frame)

        return frames

    def generateFrame(self, startFrame, transform, t):
        dim = startFrame.shape

        transformedCoordinates = np.empty([dim[0], dim[1], dim[2], 3])
        for nx in range(dim[0]):
            for ny in range(dim[1]):
                for nz in range(dim[2]):
                    transformedCoordinates[nx, ny, nz, :] = transform(np.array([nx, ny, nz]), 1)

        transformedFrame = np.empty(dim)
        for nx in range(dim[0]-1):
            for ny in range(dim[1]-1):
                for nz in range(dim[2]-1):
                    # For each tetrahedron get its bounding box and try to assign values to the integer numbered cells
                    # that fit inside the bounding box
                    for tetra in self.tetrahedra:
                        p0 = transformedCoordinates[nx + tetra[0][0], ny + tetra[0][1], nz + tetra[0][2], :]
                        p1 = transformedCoordinates[nx + tetra[1][0], ny + tetra[1][1], nz + tetra[1][2], :]
                        p2 = transformedCoordinates[nx + tetra[2][0], ny + tetra[2][1], nz + tetra[2][2], :]
                        p3 = transformedCoordinates[nx + tetra[3][0], ny + tetra[3][1], nz + tetra[3][2], :]
                        weights = [
                            startFrame[nx + tetra[0][0], ny + tetra[0][1], nz + tetra[0][2]],
                            startFrame[nx + tetra[1][0], ny + tetra[1][1], nz + tetra[1][2]],
                            startFrame[nx + tetra[2][0], ny + tetra[2][1], nz + tetra[2][2]],
                            startFrame[nx + tetra[3][0], ny + tetra[3][1], nz + tetra[3][2]]
                        ]
                        (minX, maxX, minY, maxY, minZ, maxZ) = self.getBoundingBox(p0, p1, p2, p3)
                        for bx in range(minX, maxX + 1):
                            for by in range(minY, maxY + 1):
                                for bz in range(minZ, maxZ + 1):
                                    w = self.barycentricInterpolation([p0, p1, p2, p3], weights, np.array([bx, by, bz]))
                                    transformedFrame[bx, by, bz] = w
        return transformedFrame

    def getBoundingBox(self, p0, p1, p2, p3):
        minX = int(math.ceil(min(p0[0], p1[0], p2[0], p3[0])))
        minY = int(math.ceil(min(p0[1], p1[1], p2[1], p3[1])))
        minZ = int(math.ceil(min(p0[2], p1[2], p2[2], p3[2])))
        maxX = int(math.floor(max(p0[0], p1[0], p2[0], p3[0])))
        maxY = int(math.floor(max(p0[1], p1[1], p2[1], p3[1])))
        maxZ = int(math.floor(max(p0[2], p1[2], p2[2], p3[2])))

        return minX, maxX, minY, maxY, minZ, maxZ

    def barycentricInterpolation(self, points, weights, p):
        a = points[0]
        b = points[1]
        c = points[2]
        d = points[3]

        vap = p - a
        vbp = p - b

        vab = b - a
        vac = c - a
        vad = d - a

        vbc = c - b
        vbd = d - b

        v6 = 1 / self.sctp(vab, vac, vad)
        va6 = self.sctp(vbp, vbd, vbc) * v6
        vb6 = self.sctp(vap, vac, vad) * v6
        vc6 = self.sctp(vap, vad, vab) * v6
        vd6 = self.sctp(vap, vab, vac) * v6

        return va6 * weights[0] + vb6 * weights[1] + vc6 * weights[2] + vd6 * weights[3]

    def sctp(self, a, b, c):
        return np.dot(a, np.cross(b, c))