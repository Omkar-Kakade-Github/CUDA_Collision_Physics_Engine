#include "renderer.hpp"
#include "renderer/gpu_vertex_data.hpp"
#include <cuda_runtime.h>

// Declaration of the external function from our CUDA file.
extern "C" void launchUpdateVertexArrayKernel(const PhysicObject* objects, int objectCount,
                                               CudaVertex* vertices,
                                               float textureWidth, float textureHeight,
                                               float radius);

Renderer::Renderer(PhysicsSolver& solver)
    : solver(solver), vertices(sf::Quads), d_vertices(nullptr), vertexCapacity(0)
{
    // Load texture (make sure the path is valid).
    if (!objectTexture.loadFromFile("/home/omkar/Brendan/Projects/CUDA_Physics_Collision/res/circle.png")) {
        // Handle texture loading error appropriately.
    }
    objectTexture.setSmooth(true);
}

Renderer::~Renderer() {
    if (d_vertices) {
        cudaFree(d_vertices);
    }
}

void Renderer::render(RenderContext& context) {
    int objectCount = solver.getObjectCount();
    int totalVertices = objectCount * 4;

    // Ensure the GPU vertex buffer is large enough.
    if (totalVertices > vertexCapacity) {
        if (d_vertices) cudaFree(d_vertices);
        cudaMallocManaged(&d_vertices, totalVertices * sizeof(CudaVertex));
        vertexCapacity = totalVertices;
    }

    float textureWidth = static_cast<float>(objectTexture.getSize().x);
    float textureHeight = static_cast<float>(objectTexture.getSize().y);
    float radius = 0.5f;
    
    // Launch the GPU kernel to update the vertex array.
    launchUpdateVertexArrayKernel(solver.getObjects(), objectCount, d_vertices,
                                  textureWidth, textureHeight, radius);
    
    // After the kernel completes, copy the data from the GPU buffer into our sf::VertexArray.
    // Note: d_vertices is in unified memory and can be read directly.
    vertices.resize(totalVertices);
    for (int i = 0; i < totalVertices; i++) {
        vertices[i].position = sf::Vector2f(d_vertices[i].x, d_vertices[i].y);
        vertices[i].texCoords = sf::Vector2f(d_vertices[i].u, d_vertices[i].v);
        vertices[i].color = sf::Color(d_vertices[i].color.r, d_vertices[i].color.g,
                                      d_vertices[i].color.b, d_vertices[i].color.a);
    }
    
    sf::RenderStates states;
    states.texture = &objectTexture;
    context.draw(vertices, states);
}
