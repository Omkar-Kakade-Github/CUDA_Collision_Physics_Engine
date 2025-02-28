#pragma once
#include <SFML/Graphics.hpp>
#include "physics/physics_solver.hpp"
#include "engine/window_context_handler.hpp"
#include "renderer/gpu_vertex_data.hpp"

class Renderer {
public:
    Renderer(PhysicsSolver& solver);
    ~Renderer();
    void render(RenderContext& context);
private:
    PhysicsSolver& solver;
    sf::VertexArray vertices;
    sf::Texture objectTexture;
    
    // GPU vertex buffer to hold the updated vertex data.
    CudaVertex* d_vertices;
    int vertexCapacity;  // In number of vertices.
};
