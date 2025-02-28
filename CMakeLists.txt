cmake_minimum_required(VERSION 3.18)
project(CUDA_Verlet LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SFML (make sure SFML is installed)
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# Find CUDA package (or simply set the include directory manually)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Gather all source files (both C++ and CUDA)
file(GLOB_RECURSE SRC_FILES "src/*.cpp" "src/*.cu")

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE "src")
target_link_libraries(${PROJECT_NAME} PRIVATE sfml-graphics sfml-window sfml-system)
