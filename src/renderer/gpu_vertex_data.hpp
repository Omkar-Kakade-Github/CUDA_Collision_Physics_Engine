#pragma once

// A simple color structure (matches SFML's sf::Color layout)
struct CudaColor {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
};

// A GPU‐side vertex structure (position, texture coordinate, and color)
struct CudaVertex {
    float x, y;   // Position
    float u, v;   // Texture coordinates
    CudaColor color;
};
