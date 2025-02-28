#pragma once
#include <SFML/System/Vector2.hpp>
#include "physic_object.hpp"

// Collision cell structure (grid cell with a fixed capacity)
struct CollisionCell {
    int objectsCount;
    int objects[16];
};

class PhysicsSolver {
public:
    // Constructor takes the grid (world) size and maximum number of objects.
    PhysicsSolver(sf::Vector2i worldSize, int capacity);
    ~PhysicsSolver();

    // Add an object to the simulation.
    void addObject(const PhysicObject& obj);

    // Update the simulation by dt seconds.
    void update(float dt);

    // Access simulation state for rendering.
    const PhysicObject* getObjects() const;
    int getObjectCount() const;

    // Public simulation parameters.
    sf::Vector2f gravity;
    sf::Vector2f worldSizeF;
    int subSteps; // Number of substeps for stability

private:
    int capacity;
    int objectCount;

    // Pointer to physics objects stored in unified memory.
    PhysicObject* d_objects;

    // Pointer to collision grid stored in unified memory.
    CollisionCell* d_grid;
    int gridWidth, gridHeight;

    // CUDA kernel launch helper functions (implemented in physics_solver.cu)
    void clearGrid();
    void buildGrid();
    void resolveCollisions();
    void integratePhysics(float dt);

    // Allocate and free unified memory.
    void allocateMemory();
    void freeMemory();
};
