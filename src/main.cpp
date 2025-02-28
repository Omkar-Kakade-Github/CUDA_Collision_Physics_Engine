#include "engine/window_context_handler.hpp"
#include "physics/physics_solver.hpp"
#include "renderer/renderer.hpp"

#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/Text.hpp>
#include <string>

// Entry point using the pre‑existing window context abstraction.
int main()
{
    // Create the window using the WindowContextHandler class.
    const uint32_t window_width = 1920;
    const uint32_t window_height = 1080;
    WindowContextHandler app("CUDA Verlet Simulation", sf::Vector2u(window_width, window_height), sf::Style::Default);
    RenderContext& render_context = app.getRenderContext();

    // Define simulation world size (grid dimensions) and create the physics solver.
    sf::Vector2i worldSize(1500, 1500);
    
    // The second parameter is the maximum number of objects
    PhysicsSolver solver(worldSize, 150000);

    // Create the renderer (it uses the simulation state).
    Renderer renderer(solver);

    sf::Font font;
    if (!font.loadFromFile("/home/omkar/Brendan/Projects/CUDA_Physics_Collision/res/arial.ttf")) {
        return -1;
    }

    sf::Text objectCountText;
    objectCountText.setFont(font);
    objectCountText.setCharacterSize(10);
    objectCountText.setFillColor(sf::Color::White);
    objectCountText.setPosition(1495.f, 1495.f);

    const float margin = 20.0f;
    const auto  zoom   = static_cast<float>(window_height - margin) / static_cast<float>(worldSize.y);
    render_context.setZoom(zoom);
    render_context.setFocus({worldSize.x * 0.5f, worldSize.y * 0.5f});

    // Control flag for emitting new objects.
    bool emit = true;
    
    // Toggle emission with the SPACE key.
    app.getEventManager().addKeyPressedCallback(sf::Keyboard::Space, [&](const sf::Event&) {
        emit = !emit;
    });

    constexpr float dt = 1.0f / 60.0f;
    
    // Main loop.
    while (app.run()) {
        // Emit new objects if under capacity.
        if (solver.getObjectCount() < 150000 && emit) {
            for (uint32_t i{20}; i--;) {
                float x = 2.0f - 0.2f * i;
                float y = 10.0f + 1.1f * i;
                PhysicObject obj({ x, y });
		        obj.last_position.x -= 0.2f;
                solver.addObject(obj);
            }
        }

        // Update the simulation (which launches CUDA kernels internally).
        solver.update(dt);

        // Clear the render context, render the simulation and display.
        render_context.clear();
        renderer.render(render_context);
        objectCountText.setString("Objects: " + std::to_string(solver.getObjectCount()));
        render_context.draw(objectCountText);
        render_context.display();
    }
    
    return 0;
}
