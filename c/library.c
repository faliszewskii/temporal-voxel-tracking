#include "library.h"

#include <math.h>
#include <stdlib.h>

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    int dx, dy, dz;
} Offset;

Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 vec3_add(Vec3 a, Vec3 b) {
    return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 vec3_scale(Vec3 v, float s) {
    return (Vec3){ v.x * s, v.y * s, v.z * s };
}

Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return (Vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

float vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}


enum TransformType {
    Rotate = 0,
    Pulsate = 1,
    PulsateWithNoise = 2,
};

Vec3 rotate(Vec3 coord, int t, Vec3 dim, int totalFrames) {
    Vec3 result = { coord.x * cos(t), coord.y * sin(t), coord.z };
    return result;
}

Vec3 pulsate(Vec3 x, int t, Vec3 dim, int total) {
    float pulse = powf(sinf((float)t / (total - 1) * M_PI), 2) * 0.5f + 1.0f;

    Vec3 center = { dim.x / 2.0f, dim.y / 2.0f, dim.z / 2.0f };
    Vec3 v = vec3_sub(x, center);
    Vec3 scaled = vec3_scale(v, pulse);

    return vec3_add(scaled, center);
}

Vec3 pulsateWithNoise(Vec3 coord, int t, Vec3 dim, int totalFrames) {
    float factor = 1.0f + 0.1f * t;
    Vec3 result = { coord.x * factor, coord.y * factor, coord.z * factor };
    return result;
}

typedef Vec3 (*TransformFunc)(Vec3 coord, int t, Vec3 dim, int totalFrames);

static TransformFunc transformFuncMap[] = {
    [Rotate]            = rotate,
    [Pulsate]           = pulsate,
    [PulsateWithNoise]  = pulsateWithNoise,
};

Offset tetrahedra[5][4] = {
    {{0,1,1}, {1,1,1}, {0,1,0}, {0,0,1}},
    {{0,0,1}, {1,0,1}, {1,0,0}, {1,1,1}},
    {{1,1,1}, {1,0,0}, {1,1,0}, {0,1,0}},
    {{0,0,1}, {1,0,0}, {0,0,0}, {0,1,0}},
    {{1,1,1}, {0,1,0}, {1,0,0}, {0,0,1}},
};

struct DimSize {
    int dim1;
    int dim2;
    int dim3;
};

struct DimSize createDimSize(const int x, const int y, const int z, const int w) {
    return (struct DimSize) {y * z * w, z * w, w};
}

int getIndex3d(const int x, const int y, const int z, const struct DimSize* dim) {
    return x * dim->dim2 + y * dim->dim3 + z;
}

int getIndex4d(const int x, const int y, const int z, const int w, const struct DimSize* dim) {
    return x * dim->dim1 + getIndex3d(y, z, w, dim);
}

Vec3 get_transformed_coord(TransformFunc transform, int nx, int ny, int nz, int t, int totalFrames, int x, int y, int z) {
    Vec3 coord = {nx, ny, nz};
    Vec3 dim = {x, y, z};
    return transform(coord, t, dim, totalFrames);
}

void generateFrames(float* startFrame, float* transformedFrames, int x, int y, int z, int frameCount, enum TransformType transformType, bool isCyclical) {
    const int framesToGenerate = frameCount - 1 - (isCyclical? 1: 0);
    const struct DimSize dim = createDimSize(framesToGenerate, x, y, z);
    const TransformFunc transform = transformFuncMap[transformType];
    Vec3* coords = malloc(x*y*z * sizeof(Vec3));
    // return _msize(startFrame)/ sizeof(float);

    for(int frame = 0; frame < framesToGenerate; frame++) {
        int t = frame + 1;

        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y; j++) {
                for(int k = 0; k < z; k++) {
                    coords[getIndex3d(i, j, k, &dim)] = get_transformed_coord(transform, i, j, k, t, frameCount, x, y, z);
                }
            }
        }

         for (int nx = 0; nx < x - 1; nx++) {
            for (int ny = 0; ny < y - 1; ny++) {
                for (int nz = 0; nz < z - 1; nz++) {
                    for (int ti = 0; ti < 5; ti++) {
                        Offset* tetra = tetrahedra[ti];

                        Vec3 p0 = coords[getIndex3d(nx + tetra[0].dx, ny + tetra[0].dy, nz + tetra[0].dz, &dim)];
                        Vec3 p1 = coords[getIndex3d(nx + tetra[1].dx, ny + tetra[1].dy, nz + tetra[1].dz, &dim)];
                        Vec3 p2 = coords[getIndex3d(nx + tetra[2].dx, ny + tetra[2].dy, nz + tetra[2].dz, &dim)];
                        Vec3 p3 = coords[getIndex3d(nx + tetra[3].dx, ny + tetra[3].dy, nz + tetra[3].dz, &dim)];

                        float w[4] = {
                            startFrame[getIndex3d(nx + tetra[0].dx, ny + tetra[0].dy, nz + tetra[0].dz, &dim)],
                            startFrame[getIndex3d(nx + tetra[1].dx, ny + tetra[1].dy, nz + tetra[1].dz, &dim)],
                            startFrame[getIndex3d(nx + tetra[2].dx, ny + tetra[2].dy, nz + tetra[2].dz, &dim)],
                            startFrame[getIndex3d(nx + tetra[3].dx, ny + tetra[3].dy, nz + tetra[3].dz, &dim)],
                        };

                        if (w[0] == 0 && w[1] == 0 && w[2] == 0 && w[3] == 0)
                            continue;

                        Vec3 vab = vec3_sub(p1, p0);
                        Vec3 vac = vec3_sub(p2, p0);
                        Vec3 vad = vec3_sub(p3, p0);

                        Vec3 bd_bc = vec3_cross(vec3_sub(p3, p1), vec3_sub(p2, p1));
                        Vec3 ac_ad = vec3_cross(vac, vad);
                        Vec3 ad_ab = vec3_cross(vad, vab);
                        Vec3 ab_ac = vec3_cross(vab, vac);

                        float v6 = 1.0f / vec3_dot(vab, ac_ad);

                        int minX = (int)ceil(fminf(fminf(p0.x, p1.x), fminf(p2.x, p3.x)));
                        int minY = (int)ceil(fminf(fminf(p0.y, p1.y), fminf(p2.y, p3.y)));
                        int minZ = (int)ceil(fminf(fminf(p0.z, p1.z), fminf(p2.z, p3.z)));
                        int maxX = (int)floor(fmaxf(fmaxf(p0.x, p1.x), fmaxf(p2.x, p3.x)));
                        int maxY = (int)floor(fmaxf(fmaxf(p0.y, p1.y), fmaxf(p2.y, p3.y)));
                        int maxZ = (int)floor(fmaxf(fmaxf(p0.z, p1.z), fmaxf(p2.z, p3.z)));

                        for (int bx = fmaxf(minX, 0); bx <= fminf(maxX, x - 1); bx++) {
                            for (int by = fmaxf(minY, 0); by <= fminf(maxY, y - 1); by++) {
                                for (int bz = fmaxf(minZ, 0); bz <= fminf(maxZ, z - 1); bz++) {
                                    Vec3 p = {bx, by, bz};
                                    Vec3 vap = vec3_sub(p, p0);
                                    Vec3 vbp = vec3_sub(p, p1);

                                    float va = vec3_dot(vbp, bd_bc) * v6;
                                    float vb = vec3_dot(vap, ac_ad) * v6;
                                    float vc = vec3_dot(vap, ad_ab) * v6;
                                    float vd = vec3_dot(vap, ab_ac) * v6;

                                    if (va < 0 || vb < 0 || vc < 0 || vd < 0)
                                        continue;

                                    float interp = va * w[0] + vb * w[1] + vc * w[2] + vd * w[3];
                                    transformedFrames[getIndex4d(frame, bx, by, bz, &dim)] = interp;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    free(coords);
}
