#include "physics_solver.hpp"
#include <cuda_runtime.h>
#include <cmath>

// Constants
#define CELL_CAPACITY 16
#define COLLISION_RADIUS 1.0f

// Kernel to clear the collision grid.
__global__
void clearGridKernel(CollisionCell* grid, int gridSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gridSize) {
        grid[idx].objectsCount = 0;
    }
}

// Kernel to build the grid by assigning each object to its cell.
__global__
void buildGridKernel(PhysicObject* objects, int objectCount, CollisionCell* grid,
                       int gridWidth, int gridHeight, float worldWidth, float worldHeight) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < objectCount) {
        PhysicObject obj = objects[idx];
        int cellX = (int)(obj.position.x);
        int cellY = (int)(obj.position.y);
        if (cellX >= 0 && cellX < gridWidth && cellY >= 0 && cellY < gridHeight) {
            int cellIndex = cellX + cellY * gridWidth;
            int pos = atomicAdd(&grid[cellIndex].objectsCount, 1);
            if (pos < CELL_CAPACITY) {
                grid[cellIndex].objects[pos] = idx;
            }
        }
    }
}

// Kernel to resolve collisions. Each thread processes one object and checks neighbors.
__global__
void resolveCollisionsKernel(PhysicObject* objects, CollisionCell* grid,
                             int gridWidth, int gridHeight, int objectCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= objectCount) return;
    
    // Get the object and compute its cell coordinate.
    PhysicObject obj = objects[idx];
    int cellX = (int)(obj.position.x);
    int cellY = (int)(obj.position.y);
    
    // Loop over neighboring cells.
    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            int nx = cellX + offsetX;
            int ny = cellY + offsetY;
            if (nx >= 0 && nx < gridWidth && ny >= 0 && ny < gridHeight) {
                int cellIndex = nx + ny * gridWidth;
                CollisionCell cell = grid[cellIndex];
                // Check each object in the cell.
                for (int i = 0; i < cell.objectsCount; ++i) {
                    int j = cell.objects[i];
                    // Process only pairs once.
                    if (j > idx && j < objectCount) {
                        PhysicObject obj2 = objects[j];
                        float dx = obj.position.x - obj2.position.x;
                        float dy = obj.position.y - obj2.position.y;
                        float dist2 = dx * dx + dy * dy;
                        if (dist2 < COLLISION_RADIUS * COLLISION_RADIUS && dist2 > 0.0001f) {
                            float dist = sqrtf(dist2);
                            float overlap = 0.5f * (COLLISION_RADIUS - dist);
                            float nx = dx / dist;
                            float ny = dy / dist;
                            // Apply corrections atomically.
                            atomicAdd(&objects[idx].position.x, overlap * nx);
                            atomicAdd(&objects[idx].position.y, overlap * ny);
                            atomicAdd(&objects[j].position.x, -overlap * nx);
                            atomicAdd(&objects[j].position.y, -overlap * ny);
                        }
                    }
                }
            }
        }
    }
}

// Kernel for Verlet integration. Applies gravity, damping, and boundary conditions.
__global__
void integratePhysicsKernel(PhysicObject* objects, int objectCount, float dt,
                              float damping, float gravityX, float gravityY,
                              float worldWidth, float worldHeight) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < objectCount) {
        PhysicObject obj = objects[idx];
        float dt2 = dt * dt;
        float vx = obj.position.x - obj.last_position.x;
        float vy = obj.position.y - obj.last_position.y;
        vx += (obj.acceleration.x + gravityX) * dt2;
        vy += (obj.acceleration.y + gravityY) * dt2;
        vx *= damping;
        vy *= damping;
        float newX = obj.position.x + vx;
        float newY = obj.position.y + vy;
        float margin = 2.0f;
        if (newX > worldWidth - margin) {
            newX = worldWidth - margin;
        } else if (newX < margin) {
            newX = margin;
        }

        if (newY > worldHeight - margin) {
            newY = worldHeight - margin;
        } else if (newY < margin) {
            newY = margin;
        }

        objects[idx].last_position = obj.position;
        objects[idx].position.x = newX;
        objects[idx].position.y = newY;
        objects[idx].acceleration.x = 0.0f;
        objects[idx].acceleration.y = 0.0f;
    }
}

// -------------------- PhysicsSolver methods --------------------

PhysicsSolver::PhysicsSolver(sf::Vector2i worldSize, int capacity)
    : capacity(capacity), objectCount(0)
{
    this->worldSizeF = sf::Vector2f((float)worldSize.x, (float)worldSize.y);
    this->gravity = sf::Vector2f(0.0f, 15.0f);
    this->subSteps = 4;
    gridWidth = worldSize.x;
    gridHeight = worldSize.y;
    allocateMemory();
}

PhysicsSolver::~PhysicsSolver() {
    freeMemory();
}

void PhysicsSolver::allocateMemory() {
    cudaMallocManaged(&d_objects, capacity * sizeof(PhysicObject));
    int gridSize = gridWidth * gridHeight;
    cudaMallocManaged(&d_grid, gridSize * sizeof(CollisionCell));
}

void PhysicsSolver::freeMemory() {
    cudaFree(d_objects);
    cudaFree(d_grid);
}

void PhysicsSolver::addObject(const PhysicObject& obj) {
    if (objectCount < capacity) {
        d_objects[objectCount] = obj;
        // Assign a color based on the index (for visualization)
        d_objects[objectCount].color = sf::Color((objectCount * 5) % 255,
                                                 (objectCount * 3) % 255,
                                                 (objectCount * 7) % 255);
        objectCount++;
    }
}

const PhysicObject* PhysicsSolver::getObjects() const {
    return d_objects;
}

int PhysicsSolver::getObjectCount() const {
    return objectCount;
}

void PhysicsSolver::clearGrid() {
    int gridSize = gridWidth * gridHeight;
    int threads = 256;
    int blocks = (gridSize + threads - 1) / threads;
    clearGridKernel<<<blocks, threads>>>(d_grid, gridSize);
    cudaDeviceSynchronize();
}

void PhysicsSolver::buildGrid() {
    int threads = 256;
    int blocks = (objectCount + threads - 1) / threads;
    buildGridKernel<<<blocks, threads>>>(d_objects, objectCount, d_grid,
                                           gridWidth, gridHeight,
                                           worldSizeF.x, worldSizeF.y);
    cudaDeviceSynchronize();
}

void PhysicsSolver::resolveCollisions() {
    int threads = 256;
    int blocks = (objectCount + threads - 1) / threads;
    resolveCollisionsKernel<<<blocks, threads>>>(d_objects, d_grid,
                                                   gridWidth, gridHeight,
                                                   objectCount);
    cudaDeviceSynchronize();
}

void PhysicsSolver::integratePhysics(float dt) {
    int threads = 256;
    int blocks = (objectCount + threads - 1) / threads;
    float damping = 0.999f;
    integratePhysicsKernel<<<blocks, threads>>>(d_objects, objectCount, dt, damping,
                                                 gravity.x, gravity.y,
                                                 worldSizeF.x, worldSizeF.y);
    cudaDeviceSynchronize();
}

void PhysicsSolver::update(float dt) {
    float subDt = dt / subSteps;
    for (int i = 0; i < subSteps; i++) {
        clearGrid();
        buildGrid();
        resolveCollisions();
        integratePhysics(subDt);
    }
}
