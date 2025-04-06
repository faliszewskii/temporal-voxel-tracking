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

    def generateFrames(self, startFrame, totalFrameCount, transform, isCyclical):
        frames = [startFrame]

        count = totalFrameCount
        if isCyclical:
            count -= 1

        for t in range(1, count):
            frame = self.generateFrame(startFrame, transform, t, totalFrameCount)
            frames.append(frame)

        if isCyclical:
            frames.append(startFrame)

        return frames

    def generateFrame(self, startFrame, transform, t, totalFrameCount):
        dim = startFrame.shape
        print(f"Frame {t}")

        transformedCoordinates = np.empty([dim[0], dim[1], dim[2], 3])
        for nx in range(dim[0]):
            for ny in range(dim[1]):
                for nz in range(dim[2]):
                    transformedCoordinates[nx, ny, nz, :] = transform(np.array([nx, ny, nz]), t, np.array([dim[0], dim[1], dim[2]]), totalFrameCount)

        transformedFrame = np.empty(dim)
        for nx in range(dim[0]-1):
            print(f"[{nx}, :, :]")
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

                        if weights[0] == weights[1] == weights[2] == weights[3] == 0:
                            continue

                        (minX, maxX, minY, maxY, minZ, maxZ) = self.getBoundingBox(p0, p1, p2, p3)

                        vab = p1 - p0
                        vac = p2 - p0
                        vad = p3 - p0
                        vbc = p2 - p1
                        vbd = p3 - p1
                        bd_bc = np.cross(vbd, vbc)
                        ac_ad = np.cross(vac, vad)
                        ad_ab = np.cross(vad, vab)
                        ab_ac = np.cross(vab, vac)
                        v6 = 1 / np.dot(vab, ac_ad)
                        for bx in range(max(0, minX), min(maxX + 1, dim[0])):
                            for by in range(max(0, minY), min(maxY + 1, dim[1])):
                                for bz in range(max(0, minZ), min(maxZ + 1, dim[2])):
                                    p = np.array([bx, by, bz])
                                    vap = p - p0
                                    vbp = p - p1

                                    va = np.dot(vbp, bd_bc) * v6
                                    vb = np.dot(vap, ac_ad) * v6
                                    vc = np.dot(vap, ad_ab) * v6
                                    vd = np.dot(vap, ab_ac) * v6

                                    outside = va < 0 or vb < 0 or vc < 0 or vd < 0

                                    w = va * weights[0] + vb * weights[1] + vc * weights[2] + vd * weights[3]

                                    if not outside:
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

    # def barycentricInterpolation(self, points, weights, p, vab, vac, vad, vbc, vbd, v6):
    #     vap = p - points[0]
    #     vbp = p - points[1]
    #
    #     va6 = np.dot( )self.sctp(vbp, vbd, vbc) * v6
    #     vb6 = self.sctp(vap, vac, vad) * v6
    #     vc6 = self.sctp(vap, vad, vab) * v6
    #     vd6 = self.sctp(vap, vab, vac) * v6
    #
    #     outside = va6 < 0 or vb6 < 0 or vc6 < 0 or vd6 < 0
    #
    #     return va6 * weights[0] + vb6 * weights[1] + vc6 * weights[2] + vd6 * weights[3], outside

    # def sctp(self, a, b, c):
    #     return np.dot(a, np.cross(b, c))