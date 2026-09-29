cmake_minimum_required(VERSION 3.15)
project(black_hole_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2 /fp:fast)
else()
    add_compile_options(-O3 -march=native -ffast-math -Wall -Wextra)
endif()

find_package(pybind11 REQUIRED)

pybind11_add_module(black_hole_core src/bindings.cpp)
target_include_directories(black_hole_core PRIVATE include)
