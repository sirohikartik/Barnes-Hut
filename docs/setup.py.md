# 📄 Documentation: `setup.py`

## 1. Overview & Architecture
`setup.py` provides the Python setuptools configuration for building the C++ extension `black_hole_core`. It leverages `pybind11` for binding generation and applies high-performance compiler flags (`-O3`, `-march=native`, `-ffast-math`).

---

## 2. Variables & Configuration Details

### `functions_module = Extension(...)`
Defines the C++ extension build specification:
- `name`: `'black_hole_core'` — The resulting shared library name (`black_hole_core.cpython-*.so`).
- `sources`: `['src/bindings.cpp']` — The root translation unit containing Pybind11 module bindings.
- `include_dirs`:
  - `'include'`: Local C++ header directory containing `vec3.hpp`, `body.hpp`, `octree.hpp`, and `simulation.hpp`.
  - `pybind11.get_include()`: Header path for pybind11 C++ templates.
- `language`: `'c++'` — Configures C++ compiler toolchain.
- `extra_compile_args`:
  - `-std=c++17`: Enables modern C++17 language features (`constexpr`, structured bindings).
  - `-O3`: Maximum level of compiler optimization (loop unrolling, inlining, vectorization).
  - `-march=native`: Emits instructions tuned specifically to the host processor (e.g. Apple Silicon M-series NEON instructions).
  - `-ffast-math`: Relaxes IEEE 754 compliance for floating-point arithmetic to accelerate square roots and divisions.

### `setup(...)`
Standard packaging call declaring package metadata, `ext_modules=[functions_module]`, and `zip_safe=False`.

---

## 3. Usage Commands
To build the extension in-place for local development:
```bash
python setup.py build_ext --inplace
```
