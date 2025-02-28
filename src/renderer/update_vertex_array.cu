#include <cuda_runtime.h>
#include "physics/physic_object.hpp"
#include "renderer/gpu_vertex_data.hpp"
#include <math.h>

extern "C" {

// This kernel computes, for each physics object, the four vertices of a quad.
__global__
void updateVertexArrayKernel(const PhysicObject* objects, int objectCount,
                             CudaVertex* vertices,
                             float textureWidth, float textureHeight,
                             float radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < objectCount) {
        const PhysicObject &obj = objects[idx];
        int base = idx * 4;
        float x = obj.position.x;
        float y = obj.position.y;
        
        // Compute positions for the quad vertices:
        vertices[base + 0].x = x - radius;  vertices[base + 0].y = y - radius;
        vertices[base + 1].x = x + radius;  vertices[base + 1].y = y - radius;
        vertices[base + 2].x = x + radius;  vertices[base + 2].y = y + radius;
        vertices[base + 3].x = x - radius;  vertices[base + 3].y = y + radius;
        
        // Texture coordinates (assuming the entire texture is used)
        vertices[base + 0].u = 0.0f;           vertices[base + 0].v = 0.0f;
        vertices[base + 1].u = textureWidth;    vertices[base + 1].v = 0.0f;
        vertices[base + 2].u = textureWidth;    vertices[base + 2].v = textureHeight;
        vertices[base + 3].u = 0.0f;           vertices[base + 3].v = textureHeight;
        
        // Use the object's color for all four vertices.
        vertices[base + 0].color.r = obj.color.r;
        vertices[base + 0].color.g = obj.color.g;
        vertices[base + 0].color.b = obj.color.b;
        vertices[base + 0].color.a = obj.color.a;
        // Copy the same color to all vertices.
        vertices[base + 1].color = vertices[base + 0].color;
        vertices[base + 2].color = vertices[base + 0].color;
        vertices[base + 3].color = vertices[base + 0].color;
    }
}

// Host function to launch the kernel.
void launchUpdateVertexArrayKernel(const PhysicObject* objects, int objectCount,
                                   CudaVertex* vertices,
                                   float textureWidth, float textureHeight,
                                   float radius) {
    int threads = 256;
    int blocks = (objectCount + threads - 1) / threads;
    updateVertexArrayKernel<<<blocks, threads>>>(objects, objectCount, vertices,
                                                   textureWidth, textureHeight, radius);
    cudaDeviceSynchronize();
}

} // extern "C"
