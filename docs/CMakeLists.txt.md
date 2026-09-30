# 📄 Documentation: `CMakeLists.txt`

## 1. Overview & Architecture
`CMakeLists.txt` provides an alternative build system using modern CMake (version 3.15+) for cross-platform compilation of the C++ pybind11 module `black_hole_core`.

---

## 2. Configuration & Directives

### `cmake_minimum_required(VERSION 3.15)`
Ensures modern target-based CMake functionality is available.

### `project(black_hole_sim LANGUAGES CXX)`
Defines the project name and restricts languages to C++.

### Standard & Compiler Flags
- `set(CMAKE_CXX_STANDARD 17)`: Mandates C++17.
- `set(CMAKE_CXX_STANDARD_REQUIRED ON)`: Fails configuration if the compiler does not support C++17.
- **Conditional Flags**:
  - `MSVC`: Applies `/O2 /fp:fast` for Microsoft Visual C++.
  - Other (GCC / Clang): Applies `-O3 -march=native -ffast-math -Wall -Wextra`.

### Pybind11 Integration
- `find_package(pybind11 REQUIRED)`: Locates pybind11 CMake targets and headers.
- `pybind11_add_module(black_hole_core src/bindings.cpp)`: Creates the Python extension shared library target.
- `target_include_directories(black_hole_core PRIVATE include)`: Adds `include/` to the target's include search paths.

---

## 3. Usage
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```
