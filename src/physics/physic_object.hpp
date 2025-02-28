#pragma once
#include <SFML/Graphics/Color.hpp>

// Define HOST_DEVICE to include CUDA attributes only when compiling with nvcc.
#ifdef __CUDACC__
  #define HOST_DEVICE __host__ __device__
#else
  #define HOST_DEVICE
#endif

struct Vec2 {
    float x, y;
};

struct PhysicObject {
    Vec2 position;
    Vec2 last_position;
    Vec2 acceleration;
    sf::Color color;

    HOST_DEVICE
    PhysicObject() 
      : position{0, 0}, last_position{0, 0}, acceleration{0, 0}, color(sf::Color::White) {}

    HOST_DEVICE
    PhysicObject(Vec2 pos) 
      : position(pos), last_position(pos), acceleration{0, 0}, color(sf::Color::White) {}
};
